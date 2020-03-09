import os
import lsst.log
import numpy as np
import cosmology
import astropy.io.fits as pyfits
from halo_wavelet import starlet2D
from halolet import nfwlet2D

"""
Useful functions
"""

def zMeanBin(zMin,dz,nz):
    return np.arange(zMin,zMin+dz*nz,dz)+dz/2.

def soft_thresholding(dum,thresholds):
    return np.sign(dum)*np.maximum(np.abs(dum)-thresholds,0.)

def firm_thresholding(dum,thresholds):
    mask    =   (np.abs(dum)<= thresholds)
    dum[mask]=  0.
    mask    =   (np.abs(dum)>thresholds)
    mask    =   mask&(np.abs(dum)<= 2*thresholds)
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

class massmapSparsityTask():
    def __init__(self,sources,parser):
        #file
        assert parser.has_option('file','pixDir'), 'Do not have pixDir'
        self.pixDir   =   parser.get('file','pixDir')
        assert parser.has_option('file','frameDir'), 'Do not have frameDir'
        self.frameDir =   parser.get('file','frameDir')
        assert parser.has_option('file','lbdDir'), 'Do not have lbdDir'
        self.lbdDir   =   parser.get('file','lbdDir')
        if parser.has_option('file','fieldN'):
            self.fieldN =   parser.get('file','fieldN')
        else:
            self.fieldN  =   'test'

        #sparse
        self.doDebug=   parser.getboolean('sparse','doDebug')
        self.lbd    =   parser.getfloat('sparse','lbd')
        if parser.has_option('file','lbd2'):
            self.lbd2   =   parser.getfloat('sparse','lbd2')
        else:
            self.lbd2   =   0.
        self.nframe =   parser.getint('sparse','nframe')
        self.nMax   =   parser.getint('sparse','nMax')
        self.maxR   =   parser.getint('sparse','maxR')
        outFname    =   'deltaMap_lbd%.1f_%s.fits' %(self.lbd,self.fieldN)
        self.outFname   =   os.path.join(self.lbdDir,outFname)

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
        self.nlp    =   parser.getint('lensZ','nlp')
        if self.nlp<=1:
            self.zlMin  =   0.
            self.zlscale=   1.
        else:
            self.zlMin       =   parser.getfloat('lensZ','zlMin')
            self.zlscale     =   parser.getfloat('lensZ','zlscale')
        zlBin       =   zMeanBin(self.zlMin,self.zlscale,self.nlp)

        #source z axis
        if parser.has_option('sourceZ','zname'):
            self.zname      =   parser.get('sourceZ','zname')
        else:
            self.zname      =   'z'

        self.nz     =   parser.getint('sourceZ','nz')
        if self.nz<=1:
            self.zMin   =   0.01
            self.zscale =   2.5
        else:
            self.zMin   =   parser.getfloat('sourceZ','zMin')
            self.zscale =   parser.getfloat('sourceZ','zscale')
        zsBin       =   zMeanBin(self.zMin,self.zscale,self.nz)

        self.shapeS =   (self.nz,self.ny,self.nx)
        self.shapeL =   (self.nlp,self.ny,self.nx)
        self.shapeA =   (self.nlp,self.nframe,self.ny,self.nx)

        self.cosmo  =   cosmology.Cosmo(h=1)
        lensKName   =   os.path.join(self.pixDir,'lensKernel.fits')
        lensWName   =   os.path.join(self.pixDir,'lpWeight.fits')
        if os.path.exists(lensKName):
            self.lensKernel =   pyfits.getdata(lensKName)
            self.lpWeight   =   pyfits.getdata(lensWName)
            assert self.lensKernel.shape==   (self.nz,self.nlp), 'load wrong lensing kernel'
            assert len(self.lpWeight)   ==   self.nlp,'load wroing lpWeight'
        else:
            self.lensing_kernel(zlBin,zsBin)
            pyfits.writeto(lensKName,self.lensKernel)
            pyfits.writeto(lensWName,self.lpWeight)

        dicname =   parser.get('sparse','dicname')
        if dicname=='starlet':
            self.star2D =   starlet2D(gen=2,nframe=self.nframe,ny=self.ny,nx=self.nx)
        elif dicname=='nfwlet':
            self.star2D =   nfwlet2D(nframe=self.nframe,ngrid=self.nx,smooth_scale=-1)
        self.ks2D   =   massmap_ks2D(self.ny,self.nx)

        # Read pixelized shear and mask
        g1Fname     =   os.path.join(self.pixDir,'g1Map_%s.fits'%self.fieldN)
        g2Fname     =   os.path.join(self.pixDir,'g2Map_%s.fits'%self.fieldN)
        nFname      =   os.path.join(self.pixDir,'nMap_%s.fits'%self.fieldN)
        assert os.path.exists(nFname),'cannot find pixelized shear, run binSplit first'
        self.nMap   =   pyfits.getdata(nFname)
        g1Map       =   pyfits.getdata(g1Fname)
        g2Map       =   pyfits.getdata(g2Fname)
        assert self.nMap.shape  ==   self.shapeS, 'load wrong pixelized shear'
        self.mask   =   (self.nMap>=0.1)
        self.maskF  =   (np.sum(self.nMap,axis=0)>1.)

        # Estimate mu
        if parser.has_option('sparse','mu'):
            self.mu =   parser.getfloat('sparse','mu')
        else:
            self.spectrum_norm(100)
        lsst.log.info('mu = %s' %self.mu)

        # Estimate variance plane for alpha
        sigFname    =   os.path.join(self.frameDir,'sigmaAlpha_%s.fits' %self.fieldN)
        sigma_noise =   parser.getfloat('sparse','sigma_noise')
        if os.path.exists(sigFname):
            self.sigmaA =   pyfits.getdata(sigFname)
            assert self.sigmaA.shape  ==   self.shapeA, 'load wrong noise sigma map'
        else:
            self.prox_sigmaA(100,sigma_noise)
            pyfits.writeto(sigFname,self.sigmaA)

        # Initialization
        self.alphaR =   np.zeros(self.shapeA)   # alpha
        self.deltaR =   np.zeros(self.shapeL)   # delta
        self.shearRRes   = np.zeros(self.shapeS)# shear residuals
        self.shearR =   g1Map+np.complex128(1j)*g2Map # shear
        return

    def lensing_kernel(self,zlbin,zsbin):
        lsst.log.info('Calculating lensing kernel')
        if self.nlp<=1:
            self.lensKernel =   np.ones((self.nz,self.nlp))
            self.lpWeight   =   np.ones(self.nlp)

        else:
            self.lensKernel =   np.zeros((self.nz,self.nlp))
            self.lpWeight   =   np.zeros(self.nlp)
            for i,zs in enumerate(zsbin):
                self.lensKernel[i,:]    =   self.cosmo.deltacritinv(zlbin,zs)
                self.lpWeight=  self.lpWeight+(self.lensKernel[i,:])**2.
            self.lpWeight   =   np.sqrt(self.lpWeight)
            self.lensKernel =   self.lensKernel/self.lpWeight
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

    def gradient_chi2(self):
        shearRTmp   =   self.main_forward(self.alphaR)
        self.shearRRes   =   self.shearR-shearRTmp
        dalphaR     =   -self.main_transpose(self.shearRRes)
        return dalphaR

    def gradient_TSV(self):
        dalphaR =   np.zeros(self.shapeA)
        difx    =   np.roll(self.alphaR[:,0,:,:],1,axis=-1)
        difx    =   difx-self.alphaR[:,0,:,:]
        gradx   =   np.roll(difx,-1,axis=-1)
        gradx   =   gradx-difx

        dify    =   np.roll(self.alphaR[:,0,:,:],1,axis=-2)
        dify    =   dify-self.alphaR[:,0,:,:]
        grady   =   np.roll(dify,-1,axis=-2)
        grady   =   grady-dify
        dalphaR[:,0,:,:]=(gradx+grady)/np.sqrt(20.)
        return dalphaR

    def gradient(self):
        gCh2=self.gradient_chi2()
        gTSV=self.gradient_TSV()
        return gCh2+self.lbd2*gTSV

    def spectrum_norm(self,niter):
        self.shearR=np.zeros(self.shapeS)
        norm    =   0.
        for irun in range(niter):
            # Generate a random alphaR
            np.random.seed(irun)
            self.alphaR=   np.random.randn(*self.shapeA)+np.random.random()
            """
            self.alphaR=   np.zeros(self.shapeA)
            self.alphaR[0,0,self.ny//2,self.nx//2]=1
            """
            normTmp =   np.sqrt(np.sum(self.alphaR**2.))
            self.alphaR=self.alphaR/normTmp
            dalphaT =   self.gradient()
            normTmp2=   np.sqrt(np.sum(dalphaT**2.))
            if normTmp2>norm:
                norm=normTmp2
        self.mu    = 1./(norm+1.)/1.3/(self.lbd2+1.)
        return

    def prox_sigmaA(self,niter,sigma):
        lsst.log.info('Estimating sigma map')
        outData     =   np.zeros(self.shapeA)
        if sigma<=0:
            lsst.log.info('using mock catalog to calculate alpha sigma')
            simSrcName  =   os.path.join(self.pixDir,'mock_RG_grid_%s.npy' %(self.fieldN))
            simSrc      =   np.load(simSrcName)
            for irun in range(niter):
                shearSim    =   simSrc[irun]
                alphaRSim   =   self.main_transpose(shearSim)
                outData     +=  alphaRSim**2.
            self.sigmaA     =   np.sqrt(outData/niter)*self.mu
        else:
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

        for izlp in range(self.nlp):
            for iframe in range(self.nframe):
                self.sigmaA[izlp,iframe][~self.maskF]=np.max(self.sigmaA[izlp,iframe])
        return

    def determine_thresholds(self,dalphaR):
        lbdArray=   np.ones(self.shapeA)*self.lbd
        """
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
            # the highest snr
            s1  =   snrR[iId]
            # the third highest snr?
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
        """
        self.lbdArray   =   lbdArray
        self.thresholds =   lbdArray*self.sigmaA
        return

    def reconstruct(self):
        #update deltaR
        for zl in range(self.nlp):
            alphaRZ         =   self.alphaR[zl].copy()
            #alphaRZ[0,:,:]  =   0.
            self.deltaR[zl] =   self.star2D.itransform(alphaRZ,inFou=False,outFou=False)/self.zlscale/self.lpWeight[zl]
            #self.deltaR     =   self.deltaR*self.maskF.astype(float)
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
            if (irun+1)%100==0:
                lsst.log.info('iteration: %d' %(irun))
                lsst.log.info('chi2: %.2f' %(np.sum(abs(self.shearRRes)**2.)))
                if self.doDebug:
                    self.reconstruct()
                    pyfits.writeto('deltaR_%d_%d.fits' %(iup,irun),self.deltaR.real,overwrite=True)
                    pyfits.writeto('alphaR_%d_%d.fits' %(iup,irun),self.alphaR.real,overwrite=True)
                    pyfits.writeto('lbdR_%d_%d.fits' %(iup,irun),self.lbdArray.real,overwrite=True)
        return

    def process(self,niter=1000):
        self.thresholdsMin  =   self.lbd*self.sigmaA
        threM   =   'ST'
        self.run_main_iteration(0,niter,threM)
        self.reconstruct()
        return

    def write(self):
        pyfits.writeto(self.outFname,self.deltaR.real,overwrite=True)
        return
