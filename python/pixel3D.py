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
import json
import cosmology
import haloSim
import numpy as np
from configparser import ConfigParser

C_LIGHT=2.99792458e8        # m/s
GNEWTON=6.67428e-11         # m^3/kg/s^2
KG_PER_SUN=1.98892e30       # kg/M_solar
M_PER_PARSEC=3.08568025e16  # m/pc

def four_pi_G_over_c_squared():
    """
    = 1.5*H0^2/roh_0/c^2
    We want it return 4piG/c^2 in unit of Mpc/M_solar
    in unit of m/kg
    """
    fourpiGoverc2 = 4.0*np.pi*GNEWTON/(C_LIGHT**2)
    # in unit of pc/M_solar
    fourpiGoverc2 *= KG_PER_SUN/M_PER_PARSEC
    # in unit of Mpc/M_solar
    fourpiGoverc2 /= 1.e6
    return fourpiGoverc2

class cartesianGrid3D():
    """
    pixlize in a TAN(Gnomonic)-prjected Cartesian Grid
    """
    def __init__(self,parser):
        """
        Parameters:
        -------------
        parser: parser
        transplane- unit
            unit of the parameters (arcsec/arcmin/deg)
        transplane- scale:
            pixel scale
        transplane- smooth_scale
            smoothing scale [default: -1 (no smoothing)]
        sourceZ- zbound:
            boundarys of z-axes [optional]
        sourceZ- nz:
            number of z binning
        sourceZ- zmin:
            lowest boundary of z
        sourceZ- zscale:
            z pixel scale
        """
        self.parser=parser
        # The unit of angle in the configuration
        unit=parser.get('transPlane','unit')
        # Rescaling to degree
        if unit=='degree':
            self.ratio=1.
        elif unit=='arcmin':
            self.ratio=1./60.
        elif unit=='arcsec':
            self.ratio=1./60./60.
        self.delta=parser.getfloat('transPlane','scale')*self.ratio


        ## Gaussian smoothing In the projected plane
        if parser.has_option('transPlane','smooth_scale'):
            self.sigma=parser.getfloat('transPlane','smooth_scale')*self.ratio
        else:
            self.sigma=-1

        # Line-of-sight direction for background galaxies
        if parser.has_section('sourceZ'):
            if parser.has_option('sourceZ','zbound'):
                zbound=np.array(json.loads(parser.get('sourceZ','zbound')))
                nz=len(zbound)-1
            else:
                nz=parser.getint('sourceZ','nz')
                assert nz>=1
                zmin=parser.getfloat('sourceZ','zmin')
                zmax=zmin+deltaz*(nz+0.1)
                deltaz=parser.getfloat('sourceZ','zscale')
                zbound=np.arange(zmin,zmax,deltaz)
            zcgrid=(zbound[:-1]+zbound[1:])/2.
        else:
            nz  =   1
            zbound=np.array([0.,100.])
            zcgrid= None
        self.zbound=zbound
        self.zcgrid=zcgrid
        self.nz=nz

        # Foreground plane
        if parser.has_option('lensZ','zlbound'):
            zlbound=np.array(json.loads(parser.get('sourceZ','zlbound')))
            nzl=len(zlbound)-1
        else:
            zlmin=parser.getfloat('lensZ','zlmin')
            deltazl=parser.getfloat('lensZ','zlscale')
            nzl=parser.getint('lensZ','nlp')
            assert nzl>=1
            zlmax=zlmin+deltazl*(nzl+0.1)
            zlbound=np.arange(zlmin,zlmax,deltazl)
        zlcgrid=(zlbound[:-1]+zlbound[1:])/2.
        self.zlbound=   zlbound
        self.zlcgrid=   zlcgrid
        self.nzl    =   nzl

        # For lensing kernel
        if parser.has_option('cosmology','omega_m'):
            omega_m=parser.getfloat('cosmology','omega_m')
        else:
            omega_m=0.3
        self.cosmo=cosmology.Cosmo(h=1,omega_m=omega_m)
        self.lensKernel=None
        self.pozPdfAve=None
        return

    def setupNaivePlane(self,nx,ny,xmin,ymin):
        """
        setup Tan projection plane from basic parameters
        (no rotation,no flipping)
        Parameters:
        -------------
        nx:     number of x bins [int]
        ny:     number of y bins [int]
        xmin:   minimum of x [deg]
        ymin:   minimum of y [deg]
        """
        self.nx     =   nx
        self.ny     =   ny
        dnx         =   self.nx//2
        dny         =   self.ny//2
        xmax        =   xmin+self.delta*(self.nx+0.1)
        ymax        =   ymin+self.delta*(self.ny+0.1)
        self.ra0    =   xmin+self.delta*(dnx+0.5)
        self.dec0   =   ymin+self.delta*(dny+0.5)
        self.cosdec0=   np.cos(self.dec0/180.*np.pi)
        self.sindec0=   np.sin(self.dec0/180.*np.pi)

        self.xbound= np.arange(xmin,xmax,self.delta)
        self.xcgrid= (self.xbound[:-1]+self.xbound[1:])/2.
        self.ybound= np.arange(ymin,ymax,self.delta)
        self.ycgrid= (self.ybound[:-1]+self.ybound[1:])/2.
        self.shape=(self.nz,self.ny,self.nx)
        return

    def setupTanPlane(self,ra=None,dec=None,pad=0.1,header=None):
        """
        setup Tan projection plane from (ra,dec) array or header
        (no rotation,no flipping)
        Parameters:
        -------------
        ra:     array of ra to project  [deg]
        dec:    array of dec to project [deg]
        pad:    padding distance [degree]
        header: header with projection information
        """
        if (ra is not None) and (dec is not None):
            # first try
            self.ra0    =   (np.max(ra)+np.min(ra))/2.
            self.dec0   =   (np.max(dec)+np.min(dec))/2.
            self.cosdec0=   np.cos(self.dec0/180.*np.pi)
            self.sindec0=   np.sin(self.dec0/180.*np.pi)
            x,y         =   self.project_tan(ra,dec)
            xt          =   (np.max(x)+np.min(x))/2.
            yt          =   (np.max(y)+np.min(y))/2.

            # second try
            self.ra0,self.dec0=self.iproject_tan(xt,yt)
            self.cosdec0=   np.cos(self.dec0/180.*np.pi)
            self.sindec0=   np.sin(self.dec0/180.*np.pi)
            x,y         =   self.project_tan(ra,dec)

            dxmin   =   self.ra0-(np.min(x)-pad)
            dxmax   =   (np.max(x)+pad)-self.ra0
            dnx     =   int(max(dxmin,dxmax)/self.delta+1.)
            dymin   =   self.dec0-(np.min(y)-pad)
            dymax   =   (np.max(y)+pad)-self.dec0
            dny     =   int(max(dymin,dymax)/self.delta+1.)

            # make sure we have even number of pixels in x and y
            self.nx =   2*dnx
            self.ny =   2*dny
        elif header is not None:
            assert abs(self.delta-header['CDELT1'])/self.delta<1e-2
            self.ra0    =   header['CRVAL1']
            self.dec0   =   header['CRVAL2']
            self.cosdec0=   np.cos(self.dec0/180.*np.pi)
            self.sindec0=   np.sin(self.dec0/180.*np.pi)
            self.nx     =   int(header['NAXIS1'])
            self.ny     =   int(header['NAXIS2'])
            dnx         =   self.nx//2
            dny         =   self.ny//2
        xmin    =   self.ra0-self.delta*(dnx+0.5)
        xmax    =   xmin+self.delta*(self.nx+0.1)
        ymin    =   self.dec0-self.delta*(dny+0.5)
        ymax    =   ymin+self.delta*(self.ny+0.1)

        self.xbound= np.arange(xmin,xmax,self.delta)
        self.xcgrid= (self.xbound[:-1]+self.xbound[1:])/2.
        self.ybound= np.arange(ymin,ymax,self.delta)
        self.ycgrid= (self.ybound[:-1]+self.ybound[1:])/2.
        self.shape=(self.nz,self.ny,self.nx)
        if (ra is not None) and (dec is not None):
            return x,y
        else:
            return

    def project_tan(self,ra,dec,pix=False):
        """
        TAN(Gnomonic)-prjection of sky coordiantes
        (no rotation,no flipping)
        Parameters:
        -------------
        ra:     array of ra to project  [deg]
        dec:    array of dec to project [deg]
        pix:    return in unit of pixel or not [bool]
        """
        rr      =   180.0/np.pi
        sindec  =   np.sin(dec/rr)
        cosdec  =   np.cos(dec/rr)

        capa    =   cosdec*np.cos((ra-self.ra0)/rr)
        # cos of angle distance
        cosC    =   self.sindec0*sindec+capa*self.cosdec0

        x1      =   cosdec*np.sin((ra-self.ra0)/rr)/cosC*rr+self.ra0
        x2      =   (self.cosdec0*sindec-capa*self.sindec0)/cosC*rr+self.dec0
        if pix:
            return (x1-self.xbound[0])/self.delta,(x2-self.ybound[0])/self.delta
        else:
            return x1,x2

    def iproject_tan(self,x1,x2,pix=False):
        """
        inverse TAN(Gnomonic)-prjection of pixel coordiantes
        (no rotation,no flipping)
        Parameters:
        -------------
        x1:     array of x1 pixel coord [deg]
        x2:     array of x2 pixel coord [deg]
        pix:    input in unit of pixel or not [bool]
        """

        if pix:
            # transform to in unit of degree
            x1  =   x1*self.delta+self.xbound[0]
            x2  =   x2*self.delta+self.ybound[0]

        # unit
        rr      =   180.0/np.pi
        dx      =   (x1-self.ra0)/rr
        dy      =   (x2-self.dec0)/rr

        # angle distance
        rho     =   np.sqrt(dx*dx+dy*dy)
        cc      =   np.arctan(rho)
        #
        cosC    =   np.cos(cc)
        sinC    =   np.sin(cc)

        cosP    =   dx/rho
        sinP    =   dy/rho

        xx      =   self.cosdec0*cosC-self.sindec0*sinC*sinP
        yy      =   sinC*cosP
        ra      =   self.ra0+np.arctan2(yy,xx)*rr
        dec     =   np.arcsin(self.sindec0*cosC+self.cosdec0*sinC*sinP)*rr
        return ra,dec

    def pixelize_data(self,x,y,z,v=None,ws=None,method='FFT',ave=True):
        """pixelize catalog into the cartesian grid
        Parameters:
        -------------
        x:  array
            ra of sources.
        y:  array
            dec of sources.
        z:  array
            redshifts of sources.
        v:  array
            measurements.
        ws: array [defalut: None]
            weights.
        method: string [default: FFT]
            method used to convolve with smoothing kernel
        """
        if z is None:
            # This is for 2D pixeliztion
            assert self.shape[0]==1
            z = np.ones(len(x))
        if ws is None:
            ws = np.ones(len(x))
        if self.sigma>0. and method=='sample':
            return self._pixelize_data_sample(x,y,z,v,ws)
        else:
            return self._pixelize_data_FFT(x,y,z,v,ws)

    def _pixelize_data_FFT(self,x,y,z,v=None,ws=None):
        # pixelize value field
        if v is not None:
            dataOut =   np.histogramdd((z,y,x),bins=(self.zbound,self.ybound,self.xbound),weights=v*ws)[0]
        # pixelize weight field
        weightOut=  np.histogramdd((z,y,x),bins=(self.zbound,self.ybound,self.xbound),weights=ws)[0]
        if self.sigma>0:
            # Gaussian Kernel in Fourier space
            # (normalized in configuration space)
            gausKer =   haloSim.GausAtom(ny=self.ny,nx=self.nx,sigma=self.sigma/self.delta,fou=True,lnorm=2.)
            norm    =   gausKer[0,0]
            gausKer /=  norm
            # smothing with Gausian Kernel
            # (weight and value)
            if v is not None:
                dataOut =   np.fft.ifft2(np.fft.fft2(dataOut)*gausKer).real
            weightOut=  np.fft.ifft2(np.fft.fft2(weightOut)*gausKer).real
        if v is not None:
            # avoid weight is zero
            mask            =   weightOut>0.62
            dataOut[mask]   =   dataOut[mask]/weightOut[mask]
            dataOut[~mask]  =   0.
            return dataOut
        else:
            return weightOut

    def _pixelize_data_sample(self,x,y,z,v=None,ws=None):
        xbin=   np.int_((x-self.xbound[0])/self.delta)
        ybin=   np.int_((y-self.ybound[0])/self.delta)
        if v is not None:
            dataOut=np.zeros(self.shape,v.dtype)
        else:
            dataOut=np.zeros(self.shape)
        # sample to 3 times of sigma
        rsig=   int(self.sigma/self.delta*3+1)
        for iz in range(self.shape[0]):
            if z is not None:
                mskz=   (z>=self.zbound[iz])&(z<self.zbound[iz+1])
            else:
                mskz=   np.ones(len(x),dtype=bool)
            for iy in range(self.shape[1]):
                yc  =   self.ycgrid[iy]
                msky=   mskz&(ybin>=iy-rsig)&(ybin<=iy+rsig)
                for ix in range(self.shape[2]):
                    xc  =   self.xcgrid[ix]
                    mskx=msky&(xbin>=ix-rsig)&(xbin<=ix+rsig)
                    if not  np.sum(mskx)>=1:
                        continue
                    rl2 =(x[mskx]-xc)**2.+(y[mskx]-yc)**2.
                    # convolve with Gaussian kernel
                    wg  =   np.exp(-rl2/self.sigma**2./2.)/self.sigma/np.sqrt(2.*np.pi)
                    wgsum=  np.sum(wg)
                    wl  =   wg*ws[mskx]
                    if wgsum>2./self.sigma/np.sqrt(2.*np.pi):
                        if v is not None:
                            dataOut[iz,iy,ix]=np.sum(wl*v[mskx])/np.sum(wl)
                        else:
                            dataOut[iz,iy,ix]=np.sum(wl)
        return dataOut

    def lensing_kernel(self,poz_grids=None,poz_data=None,poz_best=None,poz_ave=None,deltaIn=True):
        """Mapping from an average delta in a lens redshfit bin
        to an average kappa in a source redshift

        Parameters:
        -------------
        poz_grids:  array
            poz's bin; if None, do not include any photoz uncertainty

        poz_best:   array,len(galaxy)
            galaxy's best photoz measurements, used for galaxy binning

        poz_data:   array,(len(galaxy),len(poz_grids))
            galaxy's POZ measurements, used for deriving lensing kernel

        poz_ave:    array
            average POZ in source bins

        deltaIn:    bool [default: True (yes)]
            is mapping from (yes) delta to kappa (no) or from mass to kappa
        """

        assert (poz_data is not None)==(poz_best is not None), \
            'Please provide both photo-z bins and photo-z data'
        assert (self.nzl==1)==(self.nz==1), \
            'number of lens plane and source plane'
        if self.nzl<=1:
            return np.ones((self.nz,self.nzl))
        lensKernel =   np.zeros((self.nz,self.nzl))
        if poz_grids is None:
            # Do not use poz
            for i,zl in enumerate(self.zlcgrid):
                kl =   np.zeros(self.nz)
                mask=  (zl<self.zcgrid)
                kl[mask] =   self.cosmo.Da(zl,self.zcgrid[mask])*self.cosmo.Da(0.,zl)\
                        /self.cosmo.Da(0.,self.zcgrid[mask])
                kl*=four_pi_G_over_c_squared()
                if deltaIn:
                    # Sigma_M_zl_bin
                    # Surface masss density in lens bin
                    rhoM_ave=   self.cosmo.rho_m(zl)
                    DaBin   =   self.cosmo.Da(self.zlbound[i],self.zlbound[i+1])
                    lensKernel[:,i]=kl*rhoM_ave*DaBin
                else:
                    lensKernel[:,i]=kl*1e14
        else:
            # Use poz
            lensK =   np.zeros((len(poz_grids),self.nzl))
            for i,zl in enumerate(self.zlcgrid):
                kl=np.zeros(len(poz_grids))
                mask= (zl<poz_grids)
                #Dsl*Dl/Ds
                kl[mask] =   self.cosmo.Da(zl,poz_grids[mask])*self.cosmo.Da(0.,zl)/self.cosmo.Da(0.,poz_grids[mask])
                kl*=four_pi_G_over_c_squared()
                if deltaIn:
                    # Sigma_M_zl_bin
                    rhoM_ave=   self.cosmo.rho_m(zl)
                    DaBin   =   self.cosmo.Da(self.zlbound[i],self.zlbound[i+1])
                    lensK[:,i]= kl*rhoM_ave*DaBin
                else:
                    lensK[:,i]= kl*1e14

            if poz_ave is None:
                # Prepare the poz average
                # if it is not an input
                assert len(poz_data)==len(poz_best)
                assert len(poz_data[0])==len(poz_grids)

                # determine the average photo-z uncertainty
                poz_ave=np.zeros((self.nz,len(poz_grids)))
                for iz in range(self.nz):
                    tmp_msk=(poz_best>=self.zbound[iz])&(poz_best<self.zbound[iz+1])
                    poz_ave[iz,:]=np.average(poz_data[tmp_msk],axis=0)
            self.poz_ave=poz_ave
            lensKernel=poz_ave.dot(lensK)
            self.lensKernel=lensKernel
        return lensKernel

    def lensing_kernel_infty(self):
        """Mapping from an average delta in a lens redshfit bin
        to an average kappa in a source redshift at z=+infty
        """
        lensKernel =   np.zeros((self.nz,self.nzl))
        kl  =   self.cosmo.Da(0.,self.zlcgrid)*four_pi_G_over_c_squared()
        # Surface masss density in lens bin
        rhoM_ave=self.cosmo.rho_m(self.zlcgrid)
        DaBin=self.cosmo.Da(self.zlbound[:-1],self.zlbound[1:])
        lensKernel=kl*rhoM_ave*DaBin
        return lensKernel
