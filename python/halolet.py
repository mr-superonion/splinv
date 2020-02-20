import os
import haloSim
import numpy as np
import astropy.io.fits as pyfits

class nfwlet2D():
    """
    A Class for 2D nfwlet transform
    --------

    Construction Keywords
    -------
    nframe  :   number of frames
    ngrid   :   size of the field (pixel)
    smooth_scale:   scale radius of Gaussian smoothing kernal (pixel)
    pltDir  :   whether plot the atoms for demonstration

    Methods
    --------
    itransform:

    itranspose:


    Examples
    --------
    halolet.nfwlet2D(pltDir='plot')
    """
    def __init__(self,nframe=4,ngrid=64,smooth_scale=3,pltDir=None):
        assert ngrid%2==0,\
                'Please make sure nx and ny are even numbers'
        """
        We force ny=nx, the user should ensure that by padding 0
        """
        self.nx=ngrid
        self.ny=ngrid
        self.nframe=nframe
        self.smooth_scale=smooth_scale
        # shape of output shapelets
        self.shape2=(ngrid,ngrid)
        self.shape3=(nframe,ngrid,ngrid)
        if os.path.exists(pltDir):
            self.pltDir=pltDir
        else:
            self.pltDir=None
        self.prepareFrames()

    def prepareFrames(self):
        # Real Space
        self.fouaframes=np.zeros(self.shape3)
        self.aframes=np.zeros(self.shape3)
        # The first frame is Gaussian atom accounting for the smoothing
        self.fouaframes[0,:,:]=haloSim.GausAtom(sigma=self.smooth_scale,ngrid=self.ny,fou=True)
        self.aframes[0,:,:]=haloSim.GausAtom(sigma=self.smooth_scale,ngrid=self.ny,fou=False)

        for iframe in range(self.nframe-1):
            iAtomF=haloSim.haloCS02SigmaAtom(r_s=iframe+1,ngrid=self.ny,c=9.,smooth_scale=self.smooth_scale)
            self.fouaframes[iframe+1,:,:]=iAtomF
            self.aframes[iframe+1,:,:]=np.real(np.fft.ifft2(iAtomF))

        if self.pltDir:
            afname  =   os.path.join(self.pltDir,'nfwAtom-nframe%d.fits' %(self.nframe))
            pyfits.writeto(afname, self.aframes )
            fafname =   os.path.join(self.pltDir,'nfwAtom-Fou-nframe%d.fits' %(self.nframe))
            pyfits.writeto(fafname, self.fouaframes)
        return

    def itransform(self,dataIn,inFou=True,outFou=True):
        """
        transform from nfwlet space to image space
        """
        assert dataIn.shape==self.shape3,\
                'input should have shape (nframe,ny,nx)'
        dataOut=np.zeros(self.shape2).astype(np.complex)
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
