import numpy as np
import astropy.io.fits as pyfits
from halo_wavelet import * 

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
    def __init__(self,shearR,nMap,nframe=4,lbd=4):
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
            alphaRSim = -self.star2D.iconjugate(kappaFSim,inFou=True,outFou=False).real
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
            alphaTmp2= self.star2D.iconjugate(kappaTmp,inFou=False,outFou=False)
            normTmp2= np.sqrt(np.sum(alphaTmp2**2.))
            if normTmp2>norm:
                norm=normTmp2
        self.mu    = 1./norm/1.3
        return
        
    def gradient(self):
        shearRTmp  = self.ks2D.transform(self.kappaR,inFou=False,outFou=False)*self.mask
        shearRRes  = self.shearR-shearRTmp
        dkappaF    = self.ks2D.itransform(shearRRes,inFou=False,outFou=True)
        dalphaR    = -self.star2D.iconjugate(dkappaF,inFou=True,outFou=False).real
        return dalphaR
    
    def run_main_iteration(self,niter,doPlot=False):
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
                if doPlot:
                    pyfits.writeto('kappaR_%d.fits' %(irun),self.kappaR.real,overwrite=True)
                    pyfits.writeto('dalphaR_%d.fits' %(irun),dalphaR.real,overwrite=True)
                    pyfits.writeto('alphaR_%d.fits' %(irun),self.alphaR.real,overwrite=True)
        return
    
    def process(self):
        self.run_main_iteration(100,doPlot=False)
        return
