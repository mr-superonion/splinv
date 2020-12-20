import os
import astropy.io.fits as pyfits
import haloSim
import numpy as np
import cosmology

def zMeanBin(zMin,dz,nz):
    return np.arange(zMin,zMin+dz*nz,dz)+dz/2.

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

class nfwShearlet2D():
    """
    A Class for 2D nfwlet transform
    with different angular scale in different redshift plane
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
    def __init__(self,parser):
        ##transverse plane
        self.ny     =   parser.getint('transPlane','ny')
        self.nx     =   parser.getint('transPlane','nx')
        self.ks2D   =   massmap_ks2D(self.ny,self.nx)
        self.nframe =   parser.getint('sparse','nframe')
        self.nzl    =   parser.getint('lensZ','nlp')
        self.nzs    =   parser.getint('sourceZ','nz')
        if self.nzl  <=  1:
            self.zlMin  =   0.
            self.zlscale=   1.
        else:
            self.zlMin  =   parser.getfloat('lensZ','zlMin')
            self.zlscale=   parser.getfloat('lensZ','zlscale')
        self.zlBin      =   zMeanBin(self.zlMin,self.zlscale,self.nzl)
        self.smooth_scale =   parser.getfloat('transPlane','smooth_scale')

        # shape of output shapelets
        self.shapeP =   (self.ny,self.nx)
        self.shapeL =   (self.nzl,self.ny,self.nx)
        self.shapeA =   (self.nzl,self.nframe,self.ny,self.nx)
        self.shapeS =   (self.nzs,self.ny,self.nx)
        self.rs_base=   parser.getfloat('lensZ','rs_base')  # Mpc/h
        self.resolve_lim  =   parser.getfloat('lensZ','resolve_lim')
        if parser.has_option('cosmology','omega_m'):
            omega_m=parser.getfloat('cosmology','omega_m')
        else:
            omega_m=0.3
        self.cosmo  =   cosmology.Cosmo(h=1,omega_m=omega_m)
        lkfname     =   parser.get('prepare','lkfname')
        self.lensKernel=pyfits.getdata(lkfname)
        self.prepareFrames()

    def prepareFrames(self):
        # Initialize
        self.fouaframes =   np.zeros(self.shapeA,dtype=np.complex128)# Fourier space
        self.fouaframesDelta =   np.zeros(self.shapeA,dtype=np.complex128)# Fourier space
        self.aframes    =   np.zeros(self.shapeA,dtype=np.complex128)# Real Space
        self.rs_frame   =   -1.*np.ones((self.nzl,self.nframe)) # Radius in pixel

        for izl in range(self.nzl):
            rz      =   self.rs_base/self.cosmo.Dc(0.,self.zlBin[izl])*60.*180./np.pi
            for ifr in range(self.nframe)[::-1]:
                # For each lens redshfit bins, we begin from the
                # frame with largest angular scale radius
                rs  =   (ifr+1)*rz
                if rs<self.resolve_lim:
                    self.rs_frame[izl,ifr]=0.
                    iAtomF=haloSim.GausAtom(sigma=self.smooth_scale,ngrid=self.ny,fou=True)
                    self.fouaframesDelta[izl,ifr]=iAtomF             # Fourier Space
                    iAtomF=self.ks2D.transform(iAtomF,outFou=True)
                    self.fouaframes[izl,ifr]=iAtomF             # Fourier Space
                    self.aframes[izl,ifr]=np.fft.ifft2(iAtomF)  # Real Space
                    break
                else:
                    self.rs_frame[izl,ifr]=rs
                    iAtomF=haloSim.haloCS02SigmaAtom(r_s=rs,ngrid=self.ny,c=4.,\
                            smooth_scale=self.smooth_scale)
                    self.fouaframesDelta[izl,ifr]=iAtomF             # Fourier Space
                    iAtomF=   self.ks2D.transform(iAtomF,outFou=True)
                    self.fouaframes[izl,ifr]=iAtomF             # Fourier Space
                    self.aframes[izl,ifr]=np.fft.ifft2(iAtomF)  # Real Space
        return

    def itransformDelta(self,dataIn):
        """
        transform from nfw dictionary space to delta space
        redshift plane by redshift plane
        """
        assert dataIn.shape==self.shapeA,\
                'input should have shape (nzl,nframe,ny,nx)'

        #Initialize the output with shape (nzs,ny,nx)'
        dataOut=np.zeros(self.shapeL,dtype=np.complex128)
        for izl in range(self.nzl):
            #Initialize each lens plane with shape (ny,nx)'
            data=np.zeros(self.shapeP,dtype=np.complex128)
            for iframe in range(self.nframe)[::-1]:
                dataTmp=dataIn[izl,iframe]
                dataTmp=np.fft.fft2(dataTmp)
                data=data+(dataTmp*self.fouaframesDelta[izl,iframe])
                if self.rs_frame[izl,iframe]<self.resolve_lim:
                    # the scale of nfw halo is too small to resolve
                    # so we do not transform the next frame
                    break
            # assign the iz'th lens plane
            dataOut[izl] =   np.fft.ifft2(data)
        return dataOut

    def itransform(self,dataIn):
        """
        transform from nfw dictionary space to observations
        """
        assert dataIn.shape==self.shapeA,\
                'input should have shape (nzl,nframe,ny,nx)'

        #Initialize the output with shape (nzs,ny,nx)'
        dataOut=np.zeros(self.shapeS,dtype=np.complex128)
        for izl in range(self.nzl):
            #Initialize each lens plane with shape (ny,nx)'
            data=np.zeros(self.shapeP,dtype=np.complex128)
            for iframe in range(self.nframe)[::-1]:
                dataTmp=dataIn[izl,iframe]
                dataTmp=np.fft.fft2(dataTmp)
                data=data+(dataTmp*self.fouaframes[izl,iframe])
                if self.rs_frame[izl,iframe]<self.resolve_lim:
                    # the scale of nfw halo is too small to resolve
                    # so we do not transform the next frame
                    break
            data=np.fft.ifft2(data)
            dataOut+=data*self.lensKernel[:,izl,None,None]
        return dataOut

    def itranspose(self,dataIn):
        """
        transpose of the inverse transform operator
        """
        assert dataIn.shape==self.shapeS,\
            'input should have shape (nzs,ny,nx)'

        # Initialize the output with shape (nzl,nframe,ny,nx)
        dataOut =   np.zeros(self.shapeA,dtype=np.complex128)

        # Projection by lensing kernel
        dataIn  =   np.sum(self.lensKernel[:,:,None,None]*dataIn[:,None,:,:],axis=0)
        for izl in range(self.nzl):
            dataTmp =   np.fft.fft2(dataIn[izl])
            for iframe in range(self.nframe)[::-1]:
                # project to the dictionary space
                dataTmp2=   dataTmp*np.conjugate(self.fouaframes[izl,iframe])
                dataTmp2    =   np.fft.ifft2(dataTmp2)
                dataOut[izl,iframe,:,:]=dataTmp2
                if self.rs_frame[izl,iframe]<self.resolve_lim:
                    break
        return dataOut

class nfwlet2D():
    """
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
        assert dataIn.shape==self.shape2,\
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
