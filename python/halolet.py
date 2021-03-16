# Copyright 20200227 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
import os
import haloSim
import cosmology
import numpy as np
import astropy.io.fits as pyfits

Default_h0  =   1.

def zMeanBin(zMin,dz,nz):
    return np.arange(zMin,zMin+dz*nz,dz)+dz/2.

class massmap_ks2D():
    """
    A Class for 2D Kaiser-Squares transform
    --------

    Parameters:
    ----------
    ny,nx: number of pixels in y and x directions

    Methods:
    --------
    itransform:

    transform:
    """
    def __init__(self,ny,nx):
        self.shape   =   (ny,nx)
        self.e2phiF  =   self.__e2phiFou(self.shape)

    def __e2phiFou(self,shape):
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
        """
        K-S Transform from gamma map to kappa map
        --------

        Parameters:
        ----------
        gMap:   input gamma map
        inFou:  input in Fourier space? [default:True=yes]
        outFou: output in Fourier space? [default:True=yes]
        """
        assert gMap.shape[-2:]==self.shape
        if not inFou:
            gMap =   np.fft.fft2(gMap)
        kOMap    =   gMap/self.e2phiF*np.pi
        if not outFou:
            kOMap    =   np.fft.ifft2(kOMap)
        return kOMap

    def transform(self,kMap,inFou=True,outFou=True):
        """
        K-S Transform from kappa map to gamma map
        --------

        Parameters:
        ----------
        gMap:   input kappa map
        inFou:  input in Fourier space? [default:True=yes]
        outFou: output in Fourier space? [default:True=yes]
        """
        assert kMap.shape[-2:]==self.shape
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

    Parameters:
    ----------
    Construction Parser:
    nframe  :   number of frames
    ny,nx   :   size of the field (pixel)
    smooth_scale:   scale radius of Gaussian smoothing kernal (pixel)
    nlp     :   number of lens plane
    nz      :   number of source plane

    Methods:
    --------
    itransform: transform from halolet space to observed space

    itranspose: transpose of itransform operator

    """
    def __init__(self,parser):
        # transverse plane
        self.nframe =   parser.getint('sparse','nframe')
        self.ny     =   parser.getint('transPlane','ny')
        self.nx     =   parser.getint('transPlane','nx')
        self.ks2D   =   massmap_ks2D(self.ny,self.nx)

        # line of sight
        self.nzl    =   parser.getint('lensZ','nlp')
        self.nzs    =   parser.getint('sourceZ','nz')
        if self.nzl <=  1:
            self.zlMin  =   0.
            self.zlscale=   1.
        else:
            self.zlMin  =   parser.getfloat('lensZ','zlMin')
            self.zlscale=   parser.getfloat('lensZ','zlscale')
        self.zlBin      =   zMeanBin(self.zlMin,self.zlscale,self.nzl)
        self.smooth_scale = parser.getfloat('transPlane','smooth_scale')

        # Shape of output shapelets
        self.shapeP =   (self.ny,self.nx)                   # basic plane
        self.shapeL =   (self.nzl,self.ny,self.nx)          # lens plane
        self.shapeA =   (self.nzl,self.nframe,self.ny,self.nx) # dictionary plane
        self.shapeS =   (self.nzs,self.ny,self.nx)          # observe plane
        if parser.has_option('lensZ','atomFname'):
            atFname =   parser.get('lensZ','atomFname')
            tmp     =   pyfits.getdata(atFname)
            tmp     =   np.fft.fftshift(tmp)
            nzl,nft,nyt,nxt =   tmp.shape
            ypad    =   (self.ny-nyt)//2
            xpad    =   (self.nx-nxt)//2
            assert self.nframe==nft
            assert self.nzl==nzl
            ppad    =   ((0,0),(0,0),(ypad,ypad),(xpad,xpad))
            tmp     =   np.fft.ifftshift(np.pad(tmp,ppad))
            tmp     =   np.fft.fft2(tmp)
            self.fouaframesInter =  tmp
            self.fouaframes =   self.ks2D.transform(tmp,inFou=True,outFou=True)
            self.aframes    =   np.fft.ifft2(self.fouaframes)
        else:
            self.prepareFrames()
        lkfname     =   parser.get('prepare','lkfname')
        self.lensKernel=pyfits.getdata(lkfname)

    def prepareFrames(self):
        if parser.has_option('cosmology','omega_m'):
            omega_m =   parser.getfloat('cosmology','omega_m')
        else:
            omega_m =   0.3
        self.cosmo  =   cosmology.Cosmo(h=Default_h0,omega_m=omega_m)
        self.rs_base=   parser.getfloat('lensZ','rs_base')  # Mpc/h
        self.resolve_lim  =   parser.getfloat('lensZ','resolve_lim')
        # Initialize basis predictors
        # In configure Space
        self.aframes    =   np.zeros(self.shapeA,dtype=np.complex128)
        # In Fourier space
        self.fouaframes =   np.zeros(self.shapeA,dtype=np.complex128)
        # Intermediate basis in Fourier space
        self.fouaframesInter =   np.zeros(self.shapeA,dtype=np.complex128)
        self.rs_frame   =   -1.*np.ones((self.nzl,self.nframe)) # Radius in pixel

        for izl in range(self.nzl):
            rz      =   self.rs_base/self.cosmo.Dc(0.,self.zlBin[izl])*60.*180./np.pi
            for ifr in  range(self.nframe)[::-1]:
                # For each lens redshift bins, we begin from the
                # frame with largest angular scale radius
                rs  =   (ifr+1)*rz
                if rs<self.resolve_lim:
                    self.rs_frame[izl,ifr]= 0.
                    # l2 normalized Gaussian
                    iAtomF=haloSim.GausAtom(sigma=self.smooth_scale,ny=self.ny,nx=self.nx,fou=True)
                    self.fouaframesInter[izl,ifr]=iAtomF        # Fourier Space
                    iAtomF=self.ks2D.transform(iAtomF,inFou=True,outFou=True)
                    self.fouaframes[izl,ifr]=iAtomF             # Fourier Space
                    self.aframes[izl,ifr]=np.fft.ifft2(iAtomF)  # Configure Space
                    break
                else:
                    self.rs_frame[izl,ifr]= rs
                    # l2 normalized
                    iAtomF= haloSim.haloCS02SigmaAtom(r_s=rs,ny=self.ny,nx=self.nx,c=4.,\
                            smooth_scale=self.smooth_scale)
                    self.fouaframesInter[izl,ifr]=iAtomF        # Fourier Space
                    iAtomF= self.ks2D.transform(iAtomF,inFou=True,outFou=True)
                    # KS transform
                    self.fouaframes[izl,ifr]=iAtomF             # Fourier Space
                    self.aframes[izl,ifr]=np.fft.ifft2(iAtomF)  # Configure Space
        return

    def itransformInter(self,dataIn):
        """
        transform from model (e.g., nfwlet) dictionary space to intermediate
        (e.g., delta) space
        """
        assert dataIn.shape==self.shapeA,\
            'input should have shape (nzl,nframe,ny,nx)'

        # convolve with atom in each frame/zlens (to Fourier space)
        dataTmp =   np.fft.fft2(dataIn.astype(np.complex128),axes=(2,3))
        dataTmp =   dataTmp*self.fouaframesInter
        # sum over frames
        dataTmp =   np.sum(dataTmp,axis=1)
        # back to configure space
        dataOut =   np.fft.ifft2(dataTmp,axes=(1,2))
        return dataOut

    def itransform(self,dataIn):
        """
        transform from model (e.g., nfwlet) dictionary space to measurement
        (e.g., shear) space
        ----------
        dataIn: array to be transformed (in configure space, e.g., alpha)
        """
        assert dataIn.shape==self.shapeA,\
            'input should have shape (nzl,nframe,ny,nx)'

        # convolve with atom in each frame/zlens (to Fourier space)
        dataTmp =   np.fft.fft2(dataIn.astype(np.complex128),axes=(2,3))
        dataTmp =   dataTmp*self.fouaframes
        # sum over frames
        dataTmp2=   np.sum(dataTmp,axis=1)
        # back to configure space
        dataTmp2=   np.fft.ifft2(dataTmp2,axes=(1,2))
        # project to source plane
        dataOut =   np.sum(dataTmp2[None,:,:,:]*self.lensKernel[:,:,None,None],axis=1)
        return dataOut

    def itranspose(self,dataIn):
        """
        transpose of the inverse transform operator
        Parameters:
        ----------
        dataIn: arry to be operated (in config space, e.g., shear)
        """
        assert dataIn.shape==self.shapeS,\
            'input should have shape (nzs,ny,nx)'

        # Projection to lens plane
        # with shape=(nzl,nframe,ny,nx)
        dataTmp =   np.sum(self.lensKernel[:,:,None,None]*dataIn[:,None,:,:],axis=0)
        # Convolve with atom*
        dataTmp =   np.fft.fft2(dataTmp,axes=(1,2))
        dataTmp =   dataTmp[:,None,:,:]*np.conjugate(self.fouaframes)
        # The output with shape (nzl,nframe,ny,nx)
        dataOut =   np.fft.ifft2(dataTmp,axes=(2,3))
        return dataOut

    # def itransform(self,dataIn):
    #     """
    #     transform from nfw dictionary space to shear measurements
    #     Parameters:
    #     ----------
    #     dataIn: arry to be transformed (in config space, e.g. alpha)
    #     """
    #     assert dataIn.shape==self.shapeA,\
    #         'input should have shape (nzl,nframe,ny,nx)'

    #     # Initialize the output with shape (nzs,ny,nx)'
    #     dataOut=np.zeros(self.shapeS,dtype=np.complex128)
    #     for izl in range(self.nzl):
    #         # Initialize each lens plane with shape (ny,nx)'
    #         data=np.zeros(self.shapeP,dtype=np.complex128)
    #         for iframe in range(self.nframe)[::-1]:
    #             dataTmp=dataIn[izl,iframe]
    #             dataTmp=np.fft.fft2(dataTmp)
    #             data=data+(dataTmp*self.fouaframes[izl,iframe])
    #             if self.rs_frame[izl,iframe]<self.resolve_lim:
    #                 # The scale of nfw halo is below the resolution limit
    #                 # so we do not transform the next frame
    #                 break
    #         data=np.fft.ifft2(data)
    #         dataOut+=data*self.lensKernel[:,izl,None,None]
    #     return dataOut

    # def itranspose(self,dataIn):
    #     """
    #     transpose of the inverse transform operator
    #     Parameters:
    #     ----------
    #     dataIn: arry to be operated (in config space)
    #     """
    #     assert dataIn.shape==self.shapeS,\
    #         'input should have shape (nzs,ny,nx)'

    #     # Initialize the output with shape (nzl,nframe,ny,nx)
    #     dataOut =   np.zeros(self.shapeA,dtype=np.complex128)

    #     # Projection with lensing kernel to an array
    #     # with shape=(nzl,nframe,ny,nx)
    #     dataIn  =   np.sum(self.lensKernel[:,:,None,None]*dataIn[:,None,:,:],axis=0)
    #     for izl in range(self.nzl):
    #         dataTmp =   np.fft.fft2(dataIn[izl])
    #         for iframe in range(self.nframe)[::-1]:
    #             # Project to the dictionary space
    #             dataTmp2=   dataTmp*np.conjugate(self.fouaframes[izl,iframe])
    #             dataTmp2=   np.fft.ifft2(dataTmp2)
    #             dataOut[izl,iframe,:,:]=dataTmp2
    #             if self.rs_frame[izl,iframe]<self.resolve_lim:
    #                 # the scale of nfw halo is below the resolution limit
    #                 # so we do not transform the next frame
    #                 break
    #     return dataOut
