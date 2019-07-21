import numpy as np
import astropy.io.fits as pyfits

class starlet2D():
    def __init__(self,gen=2,nframe=4,ny=64,nx=64):
        Coeff_h0 = 3. / 8.;
        Coeff_h1 = 1. / 4.;
        Coeff_h2 = 1. / 16.;
        ker1D=np.matrix([Coeff_h2,Coeff_h1,Coeff_h0,Coeff_h1,Coeff_h2])
        self.ker2D=np.array(np.dot(ker1D.T,ker1D))
        self.nframe=nframe
        assert ny%2==0 and nx%2==0
        self.ny=ny
        self.nx=nx
        self.gen=gen
        self.shape=(nframe,ny,nx)
        self.prepareFrames()
        
    def prepareFrames(self,doPlot=False):
        self.frames=np.zeros(self.shape)
        self.aframes=np.zeros(self.shape)
        self.frames[0,0,0]=1
        self.aframes[0,0,0]=1
        if self.gen==1:
            for iframe in range(self.nframe-1):
                step=int(2**iframe+0.5)
                #first convolution
                for j in range(self.ny):
                    indexJ=(np.arange(j-2*step,j+2*step+1,step)%self.ny)[:,None]
                    for i in range(self.nx):
                        indexI=(np.arange(i-2*step,i+2*step+1,step)%self.nx)[None,:]
                        self.frames[iframe+1,j,i]=np.sum(self.ker2D*self.frames[iframe,indexJ,indexI])
                #subtraction
                self.frames[iframe,:,:]=self.frames[iframe,:,:]-self.frames[iframe+1,:,:]
                self.aframes[iframe+1,0,0]=1
        elif self.gen==2:
            for iframe in range(self.nframe-1):
                step=int(2**iframe+0.5)
                #first convolution
                for j in range(self.ny):
                    indexJ=(np.arange(j-2*step,j+2*step+1,step)%self.ny)[:,None]
                    for i in range(self.nx):
                        indexI=(np.arange(i-2*step,i+2*step+1,step)%self.nx)[None,:]
                        self.frames[iframe+1,j,i]=np.sum(self.ker2D*self.frames[iframe,indexJ,indexI])
                self.aframes[iframe+1,:,:]=self.frames[iframe+1,:,:]
                #second convolution
                dataTmp=np.empty((self.ny,self.nx))
                for j in range(self.ny):
                    indexJ=(np.arange(j-2*step,j+2*step+1,step)%self.ny)[:,None]
                    for i in range(self.nx):
                        indexI=(np.arange(i-2*step,i+2*step+1,step)%self.nx)[None,:]
                        dataTmp[j,i]=np.sum(self.ker2D*self.frames[iframe+1,indexJ,indexI])
                self.frames[iframe,:,:]=self.frames[iframe,:,:]-dataTmp
        self.fouframes=np.zeros(self.shape)
        self.fouaframes=np.zeros(self.shape)
        for iframe in range(self.nframe):
            self.fouframes[iframe,:,:]=np.fft.fft2(self.frames[iframe,:,:]).real
            self.fouaframes[iframe,:,:]=np.fft.fft2(self.aframes[iframe,:,:]).real
            
        if doPlot:
            self.framesShow=np.zeros(self.shape)
            for iframe in range(self.nframe):
                self.framesShow[iframe,:,:]=np.fft.fftshift(np.fft.ifft2(self.fouframes[iframe]).real )
            pyfits.writeto('starlet2-gen%d-nframe%d.fits' %(self.gen,self.nframe), self.frames )
            pyfits.writeto('starlet2-Fou-gen%d-nframe%d.fits' %(self.gen,self.nframe), self.framesShow)
        return
    
    def transform(self,dataIn,inFou=True,outFou=True):
        assert dataIn.shape==(self.ny,self.nx)
        if not inFou:
            dataIn2=np.fft.fft2(dataIn)
        else:
            dataIn2=dataIn.copy()
        dataOut=np.empty(self.shape)
        for iframe in range(self.nframe):
            dataTmp=dataIn2*self.fouframes[iframe,:,:]
            if not outFou:
                dataTmp=np.fft.ifft2(dataTmp).real
            dataOut[iframe,:,:]=dataTmp
        return dataOut
    
    def conjugate(self,dataIn,inFou=True,outFou=True):
        assert dataIn.shape==self.shape
        dataOut=np.zeros((self.ny,self.nx)).astype(np.complex)
        for iframe in range(self.nframe):
            dataTmp=dataIn[iframe,:,:]
            if not inFou:
                dataTmp=np.fft.fft2(dataTmp)
            dataOut=dataOut+dataTmp*self.fouframes[iframe,:,:]
        if not outFou:
            dataOut=np.fft.ifft2(dataOut)
        return dataOut.real
    
    def itransform(self,dataIn,inFou=True,outFou=True):
        assert dataIn.shape==self.shape
        dataOut=np.zeros((self.ny,self.nx)).astype(np.complex)
        for iframe in range(self.nframe):
            dataTmp=dataIn[iframe,:,:]
            if not inFou:
                dataTmp=np.fft.fft2(dataTmp)
            dataOut=dataOut+dataTmp*self.fouaframes[iframe,:,:]
        if not outFou:
            dataOut=np.fft.ifft2(dataOut).real
        return dataOut
    
    def iconjugate(self,dataIn,inFou=True,outFou=True):
        assert dataIn.shape==(self.ny,self.nx)
        dataOut=np.empty(self.shape)
        if not inFou:
            dataTmp=np.fft.fft2(dataIn)
        else:
            dataTmp=dataIn
        for iframe in range(self.nframe):
            dataTmp2=dataTmp*self.fouaframes[iframe,:,:]
            if not outFou:
                dataTmp2=np.fft.ifft2(dataTmp2).real
            dataOut[iframe,:,:]=dataTmp2
        return dataOut
