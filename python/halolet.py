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

    Construction Parser
    -------
    nframe  :   number of frames
    ny,nx   :   size of the field (pixel)
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
                    # l2 normalized gaussian
                    iAtomF=haloSim.GausAtom(sigma=self.smooth_scale,ny=self.ny,nx=self.nx,fou=True)
                    self.fouaframesDelta[izl,ifr]=iAtomF             # Fourier Space
                    iAtomF=self.ks2D.transform(iAtomF,outFou=True)
                    self.fouaframes[izl,ifr]=iAtomF             # Fourier Space
                    self.aframes[izl,ifr]=np.fft.ifft2(iAtomF)  # Real Space
                    break
                else:
                    self.rs_frame[izl,ifr]=rs
                    iAtomF=haloSim.haloCS02SigmaAtom(r_s=rs,ny=self.ny,nx=self.nx,c=4.,\
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

