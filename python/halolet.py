import os
import haloSim
import numpy as np

import astropy.io.fits as pyfits

class nfwlet2D():
    """
    halolet.nfwlet2D
    A Class for 2D nfwlet transform
    --------

    Construction Keywords
    -------
    nframe  :   number of frames
    ngrid   :   size of the field (pixel)
    smooth_scale:   scale radius of Gaussian smoothing kernal (pixel)

    Methods
    --------
    itransform: transform from halolet space to observed space

    itranspose: transpose of itransform operator


    Examples
    --------
    """
    def __init__(self,nframe=2,minframe=0,ngrid=64,smooth_scale=1.5,rs_base=2.5):
        assert ngrid%2==0,\
                'Please make sure nx and ny are even numbers'
        # We force ny=nx, the user should ensure that by padding 0
        self.nx=ngrid
        self.ny=ngrid
        self.nframe=nframe
        self.minframe=minframe
        self.smooth_scale=smooth_scale
        # shape of output shapelets
        self.shape2=(ngrid,ngrid)
        self.shape3=(nframe,ngrid,ngrid)
        self.rs_base=rs_base
        self.prepareFrames()

    def prepareFrames(self):
        # Real Space
        self.fouaframes=np.zeros(self.shape3)
        self.aframes=np.zeros(self.shape3)

        for ifr in range(self.nframe):
            iframe=ifr+self.minframe
            if iframe==0:
                # The first frame is Gaussian atom accounting for the smoothing
                self.fouaframes[ifr,:,:]=haloSim.GausAtom(sigma=self.smooth_scale,ngrid=self.ny,fou=True)
                self.aframes[ifr,:,:]=haloSim.GausAtom(sigma=self.smooth_scale,ngrid=self.ny,fou=False)
            else:
                # The other frames
                rs=iframe*self.rs_base
                iAtomF=haloSim.haloCS02SigmaAtom(r_s=rs,ngrid=self.ny,c=4.,smooth_scale=self.smooth_scale)
                self.fouaframes[ifr,:,:]=iAtomF # Fourier Space
                self.aframes[ifr,:,:]=np.real(np.fft.ifft2(iAtomF)) # Real Space

        return

    def itransform(self,dataIn,inFou=True,outFou=True):
        """
        transform from nfwlet space to image space
        """
        assert dataIn.shape==self.shape3,\
                'input should have shape (nframe,ny,nx)'
        dataOut=np.zeros(self.shape2,dtype=np.complex128)
        for iframe in range(self.nframe):
            dataTmp=dataIn[iframe,:,:]
            if not inFou:
                dataTmp=np.fft.fft2(dataTmp)
            dataOut=dataOut+dataTmp*self.fouaframes[iframe,:,:]
        if not outFou:
            dataOut=np.fft.ifft2(dataOut).real
        return dataOut

    def itranspose(self,dataIn,inFou=True,outFou=True):
        """
        transpose of the inverse transform operator
        """
        assert dataIn.shape==(self.ny,self.nx),\
                'input should have shape2 (ny,nx)'
        dataOut=np.empty(self.shape3)
        if not inFou:
            dataTmp=np.fft.fft2(dataIn)
        else:
            dataTmp=dataIn
        # the atom is fourier space should be real and symmetric
        for iframe in range(self.nframe):
            dataTmp2=dataTmp*self.fouaframes[iframe,:,:]
            if not outFou:
                dataTmp2=np.fft.ifft2(dataTmp2).real
            dataOut[iframe,:,:]=dataTmp2
        return dataOut

class starlet2D():
    """
    A Class for 2D starlet transform
    --------

    Construction Keywords
    -------
    nframe  :   number of frames
    ngrid   :   size of the field (pixel)
    smooth_scale:   scale radius of Gaussian smoothing kernal (pixel)

    Methods
    --------
    itransform:

    itranspose:

    transform:

    transpose:

    Examples
    --------
    """
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

    def prepareFrames(self):
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

    def transpose(self,dataIn,inFou=True,outFou=True):
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

    def itranspose(self,dataIn,inFou=True,outFou=True):
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
