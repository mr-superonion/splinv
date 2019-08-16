import os
import lsst.log
import numpy as np
import cosmology
import astropy.io.fits as pyfits
from halo_wavelet import * 


def zMeanBin(zMin,dz,nz):
    return np.arange(zMin,zMin+dz*nz,dz)+dz/2.
    

def soft_thresholding(dum,thresholds):
    return np.sign(dum)*np.maximum(np.abs(dum)-thresholds,0.)

def firm_thresholding(dum,thresholds):
    mask    =   (abs(dum)<= thresholds)
    dum[mask]=  0.
    mask    =   (abs(dum)>thresholds)
    mask    =   mask&(abs(dum)<= 2*thresholds)
    dum[mask]=  np.sign(dum[mask])*(2*np.abs(dum[mask])-thresholds[mask])
    return dum

def my_thresholding(dum,thresholds):
    mask    =   (abs(dum)<= thresholds)
    dum[mask]=  0.
    mask    =   (abs(dum)>thresholds)
    dum[mask]=  np.sign(dum[mask])*(abs(dum[mask])-thresholds[mask]**2./abs(dum[mask]))
    return dum



class massmap_ks2D():
    def __init__(self,ny,nx):
        self.shape   =   (ny,nx)
        self.e2phiF  =   self.e2phiFou(self.shape)

    def e2phiFou(self,shape):
        ny1,nx1 =   shape
        e2phiF  =   np.zeros(shape,dtype=complex)
        for j in range(ny1):
            jy  =   (j+ny1//2)%ny1-ny1//2
            jy  =   jy/ny1
            for i in range(nx1):
                ix  =   (i+nx1//2)%nx1-nx1//2
                ix  =   ix/nx1
                if (i**2+j**2)>0:
                    e2phiF[j,i]    =   np.complex((ix**2.-jy**2.),2.*ix*jy)/(ix**2.+jy**2.)
                else:
                    e2phiF[j,i]    =   1.
        return e2phiF*np.pi    
        
    def itransform(self,gMap,inFou=True,outFou=True):
        assert gMap.shape==self.shape
        if not inFou:
            gMap =   np.fft.fft2(gMap)
        kOMap    =   gMap/self.e2phiF*np.pi
        if not outFou:
            kOMap    =   np.fft.ifft2(kOMap)
        return kOMap

    def transform(self,kMap,inFou=True,outFou=True):
        assert kMap.shape==self.shape
        if not inFou:
            kMap =   np.fft.fft2(kMap)
        gOMap    =   kMap*self.e2phiF/np.pi
        if not outFou:
            gOMap    =   np.fft.ifft2(gOMap)
        return gOMap



class massmap_sparsity_2D():
    def __init__(self,shearR,nMap,nframe=4,lbd=4,doDebug=False):
        self.lbd   = lbd
        ny,nx      = shearR.shape
        self.shape = (nframe,ny,nx)
        self.nframe= nframe
        self.ny    = ny
        self.nx    = nx
        
        #make subtasks
        self.star2D= starlet2D(gen=2,nframe=nframe,ny=ny,nx=nx)
        self.ks2D  = massmap_ks2D(ny,nx)
        
        self.shearR= shearR
        self.nMap  = nMap
        self.mask  = (nMap>0.1).astype(np.int)
        self.sigmaA= self.prox_sigmaA(100,0.25)#np.zeros(self.shape)#
        self.doDebug=    doDebug
        if self.doDebug:
            pyfits.writeto('sigmaAlpha.fits',self.sigmaA,overwrite=True)
        
        
        self.alphaR= np.zeros(self.shape)
        self.kappaR= self.star2D.itransform(self.alphaR,inFou=False,outFou=False)
        self.spectrum_norm()
        return
    
    
    def prox_sigmaA(self,niter,sigma):
        outData=np.zeros(self.shape)
        sigMap=np.zeros((self.ny,self.nx))
        sigMap=sigma/np.sqrt(self.nMap+0.0001)*self.mask
        for irun in range(niter):
            np.random.seed(irun)
            g1Sim=np.random.randn(self.ny,self.nx)*sigMap
            g2Sim=np.random.randn(self.ny,self.nx)*sigMap
            shearSim= g1Sim+np.complex128(1j)*g2Sim
            kappaFSim = self.ks2D.itransform(shearSim,inFou=False,outFou=True)
            alphaRSim = -self.star2D.itranspose(kappaFSim,inFou=True,outFou=False).real
            outData = outData+alphaRSim**2.
        outData=np.sqrt(outData/niter)
        return outData
        
    
    def spectrum_norm(self):
        norm=0.
        for irun in range(100):
            alphaTmp= np.random.randn(self.shape[0],self.shape[1],self.shape[2])+np.random.random()*100
            normTmp = np.sqrt(np.sum(alphaTmp**2.))
            alphaTmp= alphaTmp/normTmp
            kappaTmp= self.star2D.itransform(alphaTmp,inFou=False,outFou=False)
            alphaTmp2= self.star2D.itranspose(kappaTmp,inFou=False,outFou=False)
            normTmp2= np.sqrt(np.sum(alphaTmp2**2.))
            if normTmp2>norm:
                norm=normTmp2
        self.mu    = 1./norm/1.3
        print('mu = %s' %self.mu)
        return
        
    def gradient(self):
        shearRTmp  = self.ks2D.transform(self.kappaR,inFou=False,outFou=False)*self.mask
        shearRRes  = self.shearR-shearRTmp
        dkappaF    = self.ks2D.itransform(shearRRes,inFou=False,outFou=True)
        dalphaR    = -self.star2D.itranspose(dkappaF,inFou=True,outFou=False).real
        return dalphaR
    
    def run_main_iteration(self,niter):
        tn=0
        for irun in range(niter):
            #save old kappaFou
            dalphaR = self.gradient()
            dum  = self.alphaR.real-self.mu*dalphaR.real
            dum  = np.sign(dum)*np.maximum(np.abs(dum)-self.mu*self.lbd*self.sigmaA,0.)
            #update tn and get ratio
            tnTmp= (1.+np.sqrt(1.+4*tn**2.))/2.
            ratio= (tn-1.)/tnTmp
            tn   = tnTmp
            self.alphaR=dum+(ratio*(dum-self.alphaR))
            self.alphaR[0,:,:]=0.
            #update kappaR
            self.kappaR= self.star2D.itransform(self.alphaR,inFou=False,outFou=False)
            if (irun+1)%10==0:
                print('iteration: %d' %(irun))
                if self.doDebug:
                    pyfits.writeto('kappaR_%d.fits' %(irun),self.kappaR.real,overwrite=True)
                    pyfits.writeto('dalphaR_%d.fits' %(irun),dalphaR.real,overwrite=True)
                    pyfits.writeto('alphaR_%d.fits' %(irun),self.alphaR.real,overwrite=True)
        return
    
    def process(self):
        self.run_main_iteration(100)
        return

class massmap_sparsity_3D():
    def __init__(self,sources,nframe=4,lbd=5,doDebug=False):
        self.doDebug =   doDebug
        self.lbd    =   lbd
        self.nframe =   nframe
        self.nreweight=   5
        #lens z axis
        zlMin       =   0.02
        zlscale     =   0.05
        self.nlp    =   20
        self.zlBin  =   zMeanBin(zlMin,zlscale,self.nlp)

        #transverse plane
        xMin        =   -32.
        yMin        =   -32.
        scale       =   0.5#(arcmin/pix)
        self.ny     =   128
        self.nx     =   128
        #source z axis
        self.nz     =   8
        zMin        =   0.05
        zscale      =   0.25
        if self.nz ==1:
            zMin    =   0.
            zscale  =   4.
        self.zsBin  =   zMeanBin(zMin,zscale,self.nz)
        self.shapeS =   (self.nz,self.ny,self.nx)   
        self.shapeL =   (self.nlp,self.ny,self.nx)   
        self.shapeA =   (self.nlp,nframe,self.ny,self.nx)
        
        self.cosmo  =   cosmology.Cosmo(H0=70) 
        self.lensing_kernel(zlBin,zsBin)
        
        #prepare shear and mask
        self.nMap   =   np.zeros(self.shapeS,dtype=np.int)  
        g1Map       =   np.zeros(self.shapeS)
        g2Map       =   np.zeros(self.shapeS)
        for ss in sources:
            ix  =   int((ss['ra']-xMin)//scale)
            iy  =   int((ss['dec']-yMin)//scale)
            iz  =   int((ss['z']-zMin)//zscale)
            if iz>=0 and iz<self.nz:
                g1Map[iz,iy,ix]    =   g1Map[iz,iy,ix]+ss['g1']
                g2Map[iz,iy,ix]    =   g2Map[iz,iy,ix]+ss['g2']
                self.nMap[iz,iy,ix]=   self.nMap[iz,iy,ix]+1.
        self.mask       =   (self.nMap>0.1)
        g1Map[self.mask]=   g1Map[self.mask]/self.nMap[self.mask]
        g2Map[self.mask]=   g2Map[self.mask]/self.nMap[self.mask]
        self.shearR =   g1Map+np.complex128(1j)*g2Map
        pyfits.writeto('g1Map.fits',g1Map,overwrite=True)
        pyfits.writeto('g2Map.fits',g2Map,overwrite=True)
        pyfits.writeto('nMap.fits',self.nMap,overwrite=True)
        
        #make subtasks
        self.star2D =   starlet2D(gen=2,nframe=nframe,ny=self.ny,nx=self.nx)
        self.ks2D   =   massmap_ks2D(self.ny,self.nx)
        self.spectrum_norm()
        self.prox_sigmaA(100,0.25)#np.zeros(self.shape)#
        self.alphaR=    np.zeros(self.shapeA)
        self.deltaR=    np.zeros(self.shapeL)
        return
    
    def lensing_kernel(self,zlbin,zsbin):
        self.lensKernel =   np.zeros((self.nz,self.nlp))
        self.lpWeight   =   np.zeros(self.nlp)
        for i,zs in enumerate(zsbin):
            self.lensKernel[i,:]    =   self.cosmo.deltacritinv(zlbin,zs)
            self.lpWeight=  self.lpWeight+(self.lensKernel[i,:])**2.
        self.lpWeight   =   np.sqrt(self.lpWeight)
        self.lensKernel =   self.lensKernel/self.lpWeight
        self.lpWeight   =   self.lpWeight/(zlbin[1]-zlbin[0])
        if self.doDebug:
            pyfits.writeto('lensKernel.fits',self.lensKernel,overwrite=True)
            pyfits.writeto('lpWeight.fits',self.lpWeight,overwrite=True)
        return

    def main_forward(self,alphaRIn):
        shearOut    =   np.zeros(self.shapeS).astype(np.complex128)
        for zl in range(self.nlp):
            deltaFZl=   self.star2D.itransform(alphaRIn[zl,:,:,:],inFou=False)
            shearRZl=   self.ks2D.transform(deltaFZl,outFou=False)  
            shearOut+=  (self.lensKernel[:,zl,None,None]*shearRZl)
        shearOut    =   shearOut*(self.mask.astype(np.int))
        return shearOut

    def main_transpose(self,shearRIn):
        deltaFTmp       =   np.zeros(self.shapeL).astype(np.complex128)
        for zs in range(self.nz):
            kappaFZs    =   self.ks2D.itransform(shearRIn[zs],inFou=False)
            deltaFTmp  +=   (self.lensKernel[zs,:,None,None]*kappaFZs)
        alphaRO         =   np.empty(self.shapeA)
        for zl in range(self.nlp):
            alphaRO[zl,:,:,:]=self.star2D.itranspose(deltaFTmp[zl],outFou=False).real
        return alphaRO


    def prox_sigmaA(self,niter,sigma):
        outData =   np.zeros(self.shapeA)
        sigMap  =   np.zeros(self.shapeS)
        sigMap[self.mask]  =   sigma/np.sqrt(self.nMap[self.mask])
        for irun in range(niter):
            np.random.seed(irun)
            g1Sim   =   np.random.randn(self.nz,self.ny,self.nx)*sigMap
            g2Sim   =   np.random.randn(self.nz,self.ny,self.nx)*sigMap
            shearSim=   g1Sim+np.complex128(1j)*g2Sim
            alphaRSim=  self.main_transpose(shearSim)
            outData +=  alphaRSim**2.
        self.sigmaA =   np.sqrt(outData/niter)*self.mu
        pyfits.writeto('sigmaAlpha.fits',self.sigmaA,overwrite=True)
        return outData

    def update_thresholds(self):
        mask    =   (abs(self.alphaR)>self.thresholds)
        self.thresholds[mask]= self.thresholds[mask]**2./self.alphaR[mask]
        return

    def spectrum_norm(self):
        norm=0.
        for irun in range(100):
            np.random.seed(irun)
            alphaTmp=   np.random.randn(self.nlp,self.nframe,self.ny,self.nx)+np.random.random()*100
            normTmp =   np.sqrt(np.sum(alphaTmp**2.))
            alphaTmp=   alphaTmp/normTmp
            shearTmp=   self.main_forward(alphaTmp) 
            alphaTmp2=  self.main_transpose(shearTmp) 
            normTmp2=   np.sqrt(np.sum(alphaTmp2**2.))
            if normTmp2>norm:
                norm=normTmp2
        self.mu    = 1./norm/1.3
        print('mu = %s' %self.mu)
        return
        
    def gradient(self):
        shearRTmp   =   self.main_forward(self.alphaR)   
        self.shearRRes   =   self.shearR-shearRTmp
        dalphaR     =   -self.main_transpose(self.shearRRes) 
        return dalphaR
    

    def reconstruct(self):
        #update deltaR
        for zl in range(self.nlp):
            self.deltaR[zl]= self.star2D.itransform(self.alphaR[zl],inFou=False,outFou=False)*self.lpWeight[zl]
            
        return

    def run_main_iteration(self,iup,niter,threM='ST'):
        tn=0
        for irun in range(niter):
            #save old kappaFou
            dalphaR =   -self.mu*self.gradient().real
            dum     =   self.alphaR.real+dalphaR
            if threM=='FT':
                dum  = firm_thresholding(dum,self.thresholds) 
            elif threM=='ST':
                dum  = soft_thresholding(dum,self.thresholds) 
            elif threM=='MT':
                dum  = my_thresholding(dum,self.thresholds) 
            #update tn and get ratio
            tnTmp= (1.+np.sqrt(1.+4*tn**2.))/2.
            ratio= (tn-1.)/tnTmp
            tn   = tnTmp
            self.alphaR=dum+(ratio*(dum-self.alphaR))
            self.alphaR[:,0,:,:]=0.
            if (irun+1)%20==0:
                print('iteration: %d' %(irun))
                if self.doDebug:
                    print('chi2: %.2f' %(np.sum(abs(self.shearRRes)**2.)))
                    self.reconstruct()
                    pyfits.writeto('deltaR_%d_%d.fits' %(iup,irun),self.deltaR.real,overwrite=True)
                    pyfits.writeto('dalphaR_%d_%d.fits' %(iup,irun),dalphaR.real,overwrite=True)
                    pyfits.writeto('alphaR_%d_%d.fits' %(iup,irun),self.alphaR.real,overwrite=True)
        return
    
    
    def process(self):
        self.thresholds  =   self.lbd*self.sigmaA
        for iup in range(self.nreweight):
            threM   =   'ST'
            self.run_main_iteration(iup,100,threM)
            self.update_thresholds()
        self.reconstruct()
        return

class massmap_sparsity_3D_2():
    def __init__(self,sources,parser):
        #file
        if parser.has_option('file','root'):
            self.root   =   parser.get('file','root')
        else:
            self.root   =   './'
        if parser.has_option('file','fieldN'):
            self.fieldN =   parser.get('file','fieldN')
        else:
            fieldN  =   ''
        #sparse
        self.doDebug=   parser.getboolean('sparse','doDebug')
        self.lbd    =   parser.getfloat('sparse','lbd')
        self.nframe =   parser.getint('sparse','nframe')
        self.nMax   =   parser.getint('sparse','nMax')
        self.maxR   =   parser.getint('sparse','maxR')

        #transverse plane
        if parser.has_option('transPlane','raname'):
            self.raname     =   parser.get('transPlane','raname')
        else:
            self.raname     =   'ra'
        if parser.has_option('transPlane','decname'):
            self.decname    =   parser.get('transPlane','decname')
        else:
            self.decname    =   'dec'
        self.xMin   =   parser.getfloat('transPlane','xMin')
        self.yMin   =   parser.getfloat('transPlane','yMin')
        self.scale  =   parser.getfloat('transPlane','scale')
        self.ny     =   parser.getint('transPlane'  ,'ny')
        self.nx     =   parser.getint('transPlane'  ,'nx')
        
        #lens z axis
        zlMin       =   parser.getfloat('lensZ','zlMin')
        zlscale     =   parser.getfloat('lensZ','zlscale')
        self.nlp    =   parser.getint('lensZ','nlp')
        zlBin       =   zMeanBin(zlMin,zlscale,self.nlp)
        #source z axis
        if parser.has_option('sourceZ','zname'):
            self.zname      =   parser.get('sourceZ','zname')
        else:
            self.zname      =   'z'
        self.zMin   =   parser.getfloat('sourceZ','zMin')
        self.zscale =   parser.getfloat('sourceZ','zscale')
        self.nz     =   parser.getint('sourceZ','nz')
        zsBin       =   zMeanBin(self.zMin,self.zscale,self.nz)
        
        self.shapeS =   (self.nz,self.ny,self.nx)   
        self.shapeL =   (self.nlp,self.ny,self.nx)   
        self.shapeA =   (self.nlp,self.nframe,self.ny,self.nx)
        
        self.cosmo  =   cosmology.Cosmo(H0=70) 
        lensKName   =   os.path.join(self.root,'lensKernel.fits')
        lensWName   =   os.path.join(self.root,'lpWeight.fits')
        if os.path.exists(lensKName):
            self.lensKernel =   pyfits.getdata(lensKName)
            self.lpWeight   =   pyfits.getdata(lensWName)
        else:
            self.lensing_kernel(zlBin,zsBin)
            pyfits.writeto(lensKName,self.lensKernel)
            pyfits.writeto(lensWName,self.lpWeight)
        
        #prepare shear and mask
        g1Fname     =   os.path.join(self.root,'g1Map_%s.fits'%self.fieldN)
        g2Fname     =   os.path.join(self.root,'g2Map_%s.fits'%self.fieldN)
        nFname      =   os.path.join(self.root,'nMap_%s.fits'%self.fieldN)
        if os.path.exists(nFname):
            self.nMap   =   pyfits.getdata(nFname)
            g1Map       =   pyfits.getdata(g1Fname)
            g2Map       =   pyfits.getdata(g2Fname)
            self.mask   =   (self.nMap>=0.1)
        else:
            self.nMap   =   np.zeros(self.shapeS,dtype=np.int)  
            g1Map       =   np.zeros(self.shapeS)
            g2Map       =   np.zeros(self.shapeS)
            for ss in sources:
                ix  =   int((ss[self.raname]-self.xMin)//self.scale)
                iy  =   int((ss[self.decname]-self.yMin)//self.scale)
                iz  =   int((ss[self.zname]-self.zMin)//self.zscale)
                if iz>=0 and iz<self.nz:
                    g1Map[iz,iy,ix]    =   g1Map[iz,iy,ix]+ss['g1']
                    g2Map[iz,iy,ix]    =   g2Map[iz,iy,ix]+ss['g2']
                    self.nMap[iz,iy,ix]=   self.nMap[iz,iy,ix]+1.
            self.mask       =   (self.nMap>=0.1)
            g1Map[self.mask]=   g1Map[self.mask]/self.nMap[self.mask]
            g2Map[self.mask]=   g2Map[self.mask]/self.nMap[self.mask]
            pyfits.writeto(g1Fname,g1Map)
            pyfits.writeto(g2Fname,g2Map)
            pyfits.writeto(nFname,self.nMap)
        self.shearR     =   g1Map+np.complex128(1j)*g2Map

        #make subtasks
        self.star2D =   starlet2D(gen=2,nframe=self.nframe,ny=self.ny,nx=self.nx)
        self.ks2D   =   massmap_ks2D(self.ny,self.nx)
        if parser.has_option('sparse','mu'):
            self.mu =   parser.getfloat('sparse','mu')
        else:
            self.spectrum_norm(100)
            parser.set('sparse','mu','%s' %self.mu)
        lsst.log.info('mu = %s' %self.mu)
        sigFname    =   os.path.join(self.root,'sigmaAlpha_%s.fits' %self.fieldN)
        if os.path.exists(sigFname):
            self.sigmaA =   pyfits.getdata(sigFname) 
        else:
            gsAprox     =   False
            self.prox_sigmaA(100,gsAprox)#np.zeros(self.shape)#
            pyfits.writeto(sigFname,self.sigmaA)
        self.alphaR =   np.zeros(self.shapeA)
        self.deltaR =   np.zeros(self.shapeL)
        return
    
    def lensing_kernel(self,zlbin,zsbin):
        lsst.log.info('Calculating lensing kernel')
        self.lensKernel =   np.zeros((self.nz,self.nlp))
        self.lpWeight   =   np.zeros(self.nlp)
        for i,zs in enumerate(zsbin):
            self.lensKernel[i,:]    =   self.cosmo.deltacritinv(zlbin,zs)
            self.lpWeight=  self.lpWeight+(self.lensKernel[i,:])**2.
        self.lpWeight   =   np.sqrt(self.lpWeight)
        self.lensKernel =   self.lensKernel/self.lpWeight
        self.lpWeight   =   self.lpWeight/(zlbin[1]-zlbin[0])
        return

    def main_forward(self,alphaRIn):
        shearOut    =   np.zeros(self.shapeS).astype(np.complex128)
        for zl in range(self.nlp):
            deltaFZl=   self.star2D.itransform(alphaRIn[zl,:,:,:],inFou=False)
            shearRZl=   self.ks2D.transform(deltaFZl,outFou=False)  
            shearOut+=  (self.lensKernel[:,zl,None,None]*shearRZl)
        shearOut    =   shearOut*(self.mask.astype(np.int))
        return shearOut

    def main_transpose(self,shearRIn):
        deltaFTmp       =   np.zeros(self.shapeL).astype(np.complex128)
        for zs in range(self.nz):
            kappaFZs    =   self.ks2D.itransform(shearRIn[zs],inFou=False)
            deltaFTmp  +=   (self.lensKernel[zs,:,None,None]*kappaFZs)
        alphaRO         =   np.empty(self.shapeA)
        for zl in range(self.nlp):
            alphaRO[zl,:,:,:]=self.star2D.itranspose(deltaFTmp[zl],outFou=False).real
        return alphaRO

    def spectrum_norm(self,niter):
        norm    =   0.
        for irun in range(niter):
            np.random.seed(irun)
            alphaTmp=   np.random.randn(self.nlp,self.nframe,self.ny,self.nx)+np.random.random()*100
            normTmp =   np.sqrt(np.sum(alphaTmp**2.))
            alphaTmp=   alphaTmp/normTmp
            shearTmp=   self.main_forward(alphaTmp) 
            alphaTmp2=  self.main_transpose(shearTmp) 
            normTmp2=   np.sqrt(np.sum(alphaTmp2**2.))
            if normTmp2>norm:
                norm=normTmp2
        self.mu    = 1./norm/1.3
        return
    
    
    def prox_sigmaA(self,niter,gsAprox):
        lsst.log.info('Estimating sigma map')
        outData     =   np.zeros(self.shapeA)
        if gsAprox:
            lsst.log.info('using Gaussian approximation')
            sigma   =   np.std(np.append(sources['g1'],sources['g2']))
            sigMap  =   np.zeros(self.shapeS)
            sigMap[self.mask]  =   sigma/np.sqrt(self.nMap[self.mask])
            for irun in range(niter):
                np.random.seed(irun)
                g1Sim   =   np.random.randn(self.nz,self.ny,self.nx)*sigMap
                g2Sim   =   np.random.randn(self.nz,self.ny,self.nx)*sigMap
                shearSim=   g1Sim+np.complex128(1j)*g2Sim
                alphaRSim=  self.main_transpose(shearSim)
                outData +=  alphaRSim**2.
            self.sigmaA =   np.sqrt(outData/niter)*self.mu
        else:
            lsst.log.info('using mock catalog')
            simSrcName  =   './s16aPre2D/%s_RG_mock.fits' %(self.fieldN)
            simSrc      =   pyfits.getdata(simSrcName)
            for irun in range(niter):
                raname  =   self.raname
                decname =   self.decname
                zname   =   self.zname+'_%d'  %irun
                nSim    =   np.zeros(self.shapeS)  
                g1Sim   =   np.zeros(self.shapeS)
                g2Sim   =   np.zeros(self.shapeS)
                for ss in simSrc:
                    ix  =   int((ss[raname]-self.xMin)//self.scale)
                    iy  =   int((ss[decname]-self.yMin)//self.scale)
                    iz  =   int((ss[zname]-self.zMin)//self.zscale)
                    if iz>=0 and iz<self.nz:
                        g1Sim[iz,iy,ix]    =   g1Sim[iz,iy,ix]+ss['g1_%d'%irun]
                        g2Sim[iz,iy,ix]    =   g2Sim[iz,iy,ix]+ss['g2_%d'%irun]
                        nSim[iz,iy,ix]     =   nSim[iz,iy,ix]+1.
                mask        =   (nSim>=0.1)
                g1Sim[mask] =   g1Sim[mask]/nSim[mask]
                g2Sim[mask] =   g2Sim[mask]/nSim[mask]
                shearSim    =   g1Sim+np.complex128(1j)*g2Sim
                alphaRSim   =   self.main_transpose(shearSim)
                outData     +=  alphaRSim**2.
            self.sigmaA     =   np.sqrt(outData/niter)*self.mu
        return

    def determine_thresholds(self,dalphaR):
        lbdArray=   np.ones(self.shapeA)*self.lbd
        snrR    =   np.abs(dalphaR)/self.sigmaA
        indexR  =   np.argsort(snrR,None)[::-1]
        zyxArray=   np.empty((self.nMax,3),dtype=int)
        #mask the growing region
        maskXY  =   np.ones((self.ny,self.nx),dtype=bool)
        for imax in range(self.nMax):
            itry    =   0
            while True:
                iId =   np.unravel_index(indexR[itry],self.shapeA)
                iz  =   iId[0]
                iy  =   iId[-2]
                ix  =   iId[-1]
                idY =   np.arange(iy-self.maxR,iy+self.maxR+1)[:,None]
                idX =   np.arange(ix-self.maxR,ix+self.maxR+1)[None,:]
                if maskXY[iy,ix]:
                    maskXY[idY,idX]=False
                    break
                itry+=  1
            s1  =   snrR[iId]
            s2  =   np.sort(snrR[:,:,idY,idX],None)[-2]
            lbdArray[:,:,idY,idX]=np.maximum(lbdArray[:,:,idY,idX],(s1+s2)/2.)
            self.minSnr =  min((s1+s2)/2.,self.minSnr)
            zyxArray[imax,:]=np.array([iz,iy,ix])
        lsst.log.info('minSnr: %s ' %self.minSnr)
        lbdArray    =   np.maximum(lbdArray,self.minSnr)
        #only grow in one redshift plane
        for imax in range(self.nMax):
            iz  =   zyxArray[imax,0]
            iy  =   zyxArray[imax,1]
            ix  =   zyxArray[imax,2]
            idY =   np.arange(iy-self.maxR,iy+self.maxR+1)[:,None]
            idX =   np.arange(ix-self.maxR,ix+self.maxR+1)[None,:]
            lbdArray[iz,:,idY,idX]=self.lbd
        self.lbdArray   =   lbdArray  
        self.thresholds =   lbdArray*self.sigmaA
        return

        
    def gradient(self):
        shearRTmp   =   self.main_forward(self.alphaR)   
        self.shearRRes   =   self.shearR-shearRTmp
        dalphaR     =   -self.main_transpose(self.shearRRes) 
        return dalphaR
    

    def reconstruct(self):
        #update deltaR
        for zl in range(self.nlp):
            self.deltaR[zl]= self.star2D.itransform(self.alphaR[zl],inFou=False,outFou=False)*self.lpWeight[zl]
            
        return

    def run_main_iteration(self,iup,niter,threM='FT'):
        tn=0
        self.minSnr  =   10000.
        for irun in range(niter):
            #save old kappaFou
            dalphaR =   -self.mu*self.gradient().real
            if self.minSnr>self.lbd:
                self.determine_thresholds(dalphaR)
            dum     =   self.alphaR.real+dalphaR
            if threM=='FT':
                dum  = firm_thresholding(dum,self.thresholds) 
            elif threM=='ST':
                dum  = soft_thresholding(dum,self.thresholds) 
            elif threM=='MT':
                dum  = my_thresholding(dum,self.thresholds) 
            #update tn and get ratio
            tnTmp= (1.+np.sqrt(1.+4*tn**2.))/2.
            ratio= (tn-1.)/tnTmp
            tn   = tnTmp
            self.alphaR=dum+(ratio*(dum-self.alphaR))
            #self.alphaR[:,0,:,:]=0.
            if (irun+1)%50==0:
                lsst.log.info('iteration: %d' %(irun))
                if self.doDebug:
                    lsst.log.info('chi2: %.2f' %(np.sum(abs(self.shearRRes)**2.)))
                    self.reconstruct()
                    pyfits.writeto('deltaR_%d_%d.fits' %(iup,irun),self.deltaR.real,overwrite=True)
                    pyfits.writeto('alphaR_%d_%d.fits' %(iup,irun),self.alphaR.real,overwrite=True)
                    pyfits.writeto('lbdR_%d_%d.fits' %(iup,irun),self.lbdArray.real,overwrite=True)
        return
    
    def process(self,niter=1000):
        self.thresholdsMin  =   self.lbd*self.sigmaA
        threM   =   'FT'
        self.run_main_iteration(0,niter,threM)
        self.reconstruct()
        return

    def write(self):
        return
