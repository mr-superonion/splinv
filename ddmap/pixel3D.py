# Copyright 20211226 Xiangchong Li.
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
import json
import numpy as np
from .default import *
from .import halosim
from astropy.cosmology import FlatLambdaCDM as Cosmo

class cartesianGrid3D():
    """
    pixlize in a TAN(Gnomonic)-prjected Cartesian Grid
    Parameters:
        parser: parser
    """
    def __init__(self,parser):
        # The unit of angle in the configuration
        unit    =   parser.get('transPlane','unit')
        # Rescaling to degree
        if unit ==  'degree':
            self.ratio= 1.
        elif unit== 'arcmin':
            self.ratio= 1./60.
        elif unit== 'arcsec':
            self.ratio= 1./60./60.
        self.scale= parser.getfloat('transPlane','scale')*self.ratio

        # Line-of-sight direction for background galaxies
        if parser.has_section('sources'):
            if parser.has_option('sources','zbound'):
                zbound=np.array(json.loads(parser.get('sources','zbound')))
                nz  =   len(zbound)-1
            else:
                nz  =   parser.getint('sources','nz')
                assert nz>=1
                zmin=   parser.getfloat('sources','zmin')
                zmax=   zmin+deltaz*(nz+0.1)
                deltaz= parser.getfloat('sources','zscale')
                zbound= np.arange(zmin,zmax,deltaz)
            zcgrid  =   (zbound[:-1]+zbound[1:])/2.
        else:
            nz  =   1
            zbound  =   np.array([0.,100.])
            zcgrid  =   None
        self.zbound =   zbound
        self.zcgrid =   zcgrid
        self.nz =   nz

        # Transverse Plane
        if parser.has_option('transPlane','nx')\
            and parser.has_option('transPlane','xmin'):
            nx      =   parser.getint('transPlane','nx')
            ny      =   parser.getint('transPlane','ny')
            xmin    =   parser.getfloat('transPlane','xmin')*self.ratio
            ymin    =   parser.getfloat('transPlane','ymin')*self.ratio
            self.setupNaivePlane(nx,ny,xmin,ymin)

        ## Gaussian smoothing in the transverse plane
        if parser.has_option('transPlane','smooth_scale'):
            self.sigma  =   parser.getfloat('transPlane','smooth_scale')\
                            *self.ratio
        else:
            self.sigma  =   -1
        self.sigma_pix  =   self.sigma/self.scale

        # Foreground plane
        if parser.has_option('lens','zlbound'):
            zlbound=np.array(json.loads(parser.get('sources','zlbound')))
            nzl=len(zlbound)-1
        else:
            zlmin=parser.getfloat('lens','zlmin')
            deltazl=parser.getfloat('lens','zlscale')
            nzl=parser.getint('lens','nlp')
            assert nzl>=1
            zlmax=zlmin+deltazl*(nzl+0.1)
            zlbound=np.arange(zlmin,zlmax,deltazl)
        zlcgrid=(zlbound[:-1]+zlbound[1:])/2.
        self.zlbound=   zlbound
        self.zlcgrid=   zlcgrid
        self.nzl    =   nzl

        # For lensing kernel
        if parser.has_option('cosmology','omega_m'):
            omega_m =   parser.getfloat('cosmology','omega_m')
        else:
            omega_m =   0.3
        self.cosmo=Cosmo(H0=Default_h0*100.,Om0=omega_m)
        self.lensKernel =   None
        self.pozPdfAve  =   None
        return

    def setupNaivePlane(self,nx,ny,xmin,ymin):
        """setup Tan projection plane from basic parameters
        (no rotation,no flipping)
        Parameters:
            nx:     number of x bins [int]
            ny:     number of y bins [int]
            xmin:   minimum of x [deg]
            ymin:   minimum of y [deg]
        """
        self.nx     =   nx
        self.ny     =   ny
        dnx         =   self.nx//2
        dny         =   self.ny//2
        xmax        =   xmin+self.scale*(self.nx+0.1)
        ymax        =   ymin+self.scale*(self.ny+0.1)
        self.ra0    =   xmin+self.scale*(dnx+0.5)
        self.dec0   =   ymin+self.scale*(dny+0.5)
        self.cosdec0=   np.cos(self.dec0/180.*np.pi)
        self.sindec0=   np.sin(self.dec0/180.*np.pi)

        self.xbound =   np.arange(xmin,xmax,self.scale)
        self.xcgrid =   (self.xbound[:-1]+self.xbound[1:])/2.
        self.ybound =   np.arange(ymin,ymax,self.scale)
        self.ycgrid =   (self.ybound[:-1]+self.ybound[1:])/2.
        self.shape  =   (self.nz,self.ny,self.nx)
        return

    def setupTanPlane(self,ra=None,dec=None,pad=0.1,header=None):
        """setup Tan projection plane from (ra,dec) array or header
        (no rotation,no flipping)
        Parameters:
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
            dnx     =   int(max(dxmin,dxmax)/self.scale+1.)
            dymin   =   self.dec0-(np.min(y)-pad)
            dymax   =   (np.max(y)+pad)-self.dec0
            dny     =   int(max(dymin,dymax)/self.scale+1.)

            # make sure we have even number of pixels in x and y
            self.nx =   2*dnx
            self.ny =   2*dny
        elif header is not None:
            assert abs(self.scale-header['CDELT1'])/self.scale<1e-2
            self.ra0    =   header['CRVAL1']
            self.dec0   =   header['CRVAL2']
            self.cosdec0=   np.cos(self.dec0/180.*np.pi)
            self.sindec0=   np.sin(self.dec0/180.*np.pi)
            self.nx     =   int(header['NAXIS1'])
            self.ny     =   int(header['NAXIS2'])
            dnx         =   self.nx//2
            dny         =   self.ny//2
        else:
            raise ValueError('should input (ra,dec) or header')

        xmin    =   self.ra0-self.scale*(dnx+0.5)
        xmax    =   xmin+self.scale*(self.nx+0.1)
        ymin    =   self.dec0-self.scale*(dny+0.5)
        ymax    =   ymin+self.scale*(self.ny+0.1)

        self.xbound= np.arange(xmin,xmax,self.scale)
        self.xcgrid= (self.xbound[:-1]+self.xbound[1:])/2.
        self.ybound= np.arange(ymin,ymax,self.scale)
        self.ycgrid= (self.ybound[:-1]+self.ybound[1:])/2.
        self.shape=(self.nz,self.ny,self.nx)
        if (ra is not None) and (dec is not None):
            return x,y
        else:
            return

    def project_tan(self,ra,dec,pix=False):
        """TAN(Gnomonic)-prjection of sky coordiantes
        (no rotation,no flipping)
        Parameters:
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
            return (x1-self.xbound[0])/self.scale,(x2-self.ybound[0])/self.scale
        else:
            return x1,x2

    def iproject_tan(self,x1,x2,pix=False):
        """inverse TAN(Gnomonic)-prjection of pixel coordiantes
        (no rotation,no flipping)
        Parameters:
            x1:     array of x1 pixel coord [deg]
            x2:     array of x2 pixel coord [deg]
            pix:    input in unit of pixel or not [bool]
        """

        if pix:
            # transform to in unit of degree
            x1  =   x1*self.scale+self.xbound[0]
            x2  =   x2*self.scale+self.ybound[0]

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

    def pixelize_data(self,x,y,z,v=None,ws=None,method='FFT'):
        """pixelize catalog into the cartesian grid
        Parameters:
            x:  array
                ra of sources. (deg)
            y:  array
                dec of sources. (deg)
            z:  array
                redshifts of sources.
            v:  array
                measurements. (e.g., shear, kappa)
            ws: array [defalut: None]
                weights.
            method: string [default: 'FFT']
                method to convolve with the Gaussian kernel
        """
        if z is None:
            # This is for 2D pixeliztion
            assert self.shape[0]==1
            z = np.ones(len(x))
        if ws is None:
            # Without weight
            ws = np.ones(len(x))
        if method=='sample':
            if self.sigma>0.:
                return self._pixelize_data_sample(x,y,z,v,ws)
            else:
                raise ValueError("when method is 'sample', smooth_scale must >0")
        elif method=='FFT':
            return self._pixelize_data_FFT(x,y,z,v,ws)
        else:
            raise ValueError("method must be 'FFT' or 'sample'")

    def _pixelize_data_FFT(self,x,y,z,v=None,ws=None):
        if v is not None:
            # pixelize value field
            dataOut =   np.histogramdd((z,y,x),bins=(self.zbound,self.ybound,self.xbound),weights=v*ws)[0]
        else:
            dataOut =   []
        # pixelize weight field
        weightOut=  np.histogramdd((z,y,x),bins=(self.zbound,self.ybound,self.xbound),weights=ws)[0]
        if self.sigma_pix>0.01:
            # Gaussian Kernel in Fourier space
            # (normalize to flux-1)
            gausKer =   halosim.GausAtom(ny=self.ny,nx=self.nx,sigma=self.sigma_pix,fou=True)
            norm    =   gausKer[0,0]
            gausKer /=  norm
            # smothing with Gausian Kernel
            # (for both weight and value)
            weightOut=  np.fft.ifft2(np.fft.fft2(weightOut)*gausKer).real
            if v is not None:
                dataOut=np.fft.ifft2(np.fft.fft2(dataOut)*gausKer).real

        if v is not None:
            # avoid weight is zero
            mask            =   weightOut>0.1
            dataOut[mask]   =   dataOut[mask]/weightOut[mask]
            dataOut[~mask]  =   0.
            return dataOut
        else:
            return weightOut

    def _pixelize_data_sample(self,x,y,z,v=None,ws=None):
        xbin=   np.int_((x-self.xbound[0])/self.scale)
        ybin=   np.int_((y-self.ybound[0])/self.scale)
        if v is not None:
            dataOut=np.zeros(self.shape,v.dtype)
        else:
            dataOut=np.zeros(self.shape)
        # sample to 3 times of sigma
        rsig=   int(self.sigma_pix*3+1)
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
                    if wgsum>0.1/self.sigma/np.sqrt(2.*np.pi):
                        if v is not None:
                            dataOut[iz,iy,ix]=np.sum(wl*v[mskx])/np.sum(wl)
                        else:
                            dataOut[iz,iy,ix]=np.sum(wl)
        return dataOut

    def lensing_kernel(self,poz_grids=None,poz_data=None,poz_best=None,z_dens=None,deltaIn=True):
        """ Lensing kernel

        Parameters:
            poz_grids:  array
                poz's bin; if None, do not include any photoz uncertainty
            poz_best:   array, in shape of len(galaxy)
                galaxy's best photoz measurements, used for galaxy binning
            poz_data:   2D array, in shape of (len(galaxy), len(poz_grids))
                galaxy's POZ measurements, used for deriving lensing kernel
            z_dens:    array
                average POZ in source bins
            deltaIn:    bool [default: True]
                is mapping from delta to kappa (True) or from mass to kappa (False)
        Returns:
            lensing kernel
        """

        assert (poz_data is not None)==(poz_best is not None), \
            'Please provide both photo-z bins and photo-z data'
        assert (self.nzl==1)==(self.nz==1), \
            'number of lens plane and source plane'
        if self.nzl<=1:
            return np.ones((self.nz,self.nzl))
        lensKernel =   np.zeros((self.nz,self.nzl))
        if poz_grids is None:
            # Without z_density or photoz posterier
            for i,zl in enumerate(self.zlcgrid):
                kl =   np.zeros(self.nz)
                mask=  (zl<self.zcgrid)
                kl[mask] =   self.cosmo.angular_diameter_distance_z1z2(zl,self.zcgrid[mask]).value\
                    *self.cosmo.angular_diameter_distance_z1z2(0.,zl).value\
                    /self.cosmo.angular_diameter_distance_z1z2(0.,self.zcgrid[mask]).value
                kl*=    four_pi_G_over_c_squared()
                if deltaIn:
                    # When the input is density contrast
                    # Surface masss density in lens bin
                    rhoM_ave=   self.cosmo.critical_density(zl).to_value(unit=rho_unt)*self.cosmo.Om(zl)
                    DaBin   =   self.cosmo.angular_diameter_distance_z1z2(self.zlbound[i],self.zlbound[i+1]).value
                    lensKernel[:,i]=kl*rhoM_ave*DaBin
                else:
                    # When the input is M_200/1e14
                    lensKernel[:,i]=kl*1e14
        else:
            # with z_dens or photo-z posterier
            lensK =   np.zeros((len(poz_grids),self.nzl))
            for i,zl in enumerate(self.zlcgrid):
                kl=np.zeros(len(poz_grids))
                mask= (zl<poz_grids)
                #Dsl*Dl/Ds
                kl[mask] =   self.cosmo.angular_diameter_distance_z1z2(zl,poz_grids[mask]).value\
                    *self.cosmo.angular_diameter_distance_z1z2(0.,zl).value\
                    /self.cosmo.angular_diameter_distance_z1z2(0.,poz_grids[mask]).value
                kl*=    four_pi_G_over_c_squared()
                if deltaIn:
                    # Sigma_M_zl_bin
                    rhoM_ave=   self.cosmo.critical_density(zl).to_value(unit=rho_unt)*self.cosmo.Om(zl)
                    DaBin   =   self.cosmo.angular_diameter_distance_z1z2(self.zlbound[i],self.zlbound[i+1]).value
                    lensK[:,i]= kl*rhoM_ave*DaBin
                else:
                    lensK[:,i]= kl*1e14
            if z_dens is None:
                # Prepare the n(z) if it is not an input.
                # using the stacked posterier of photo-z
                assert len(poz_data)==len(poz_best)
                assert len(poz_data[0])==len(poz_grids)
                # determine the average photo-z posterier
                z_dens=np.zeros((self.nz,len(poz_grids)))
                for iz in range(self.nz):
                    tmp_msk     =   (poz_best>=self.zbound[iz])&(poz_best<self.zbound[iz+1])
                    z_dens[iz,:]=   np.average(poz_data[tmp_msk],axis=0)

            self.z_dens =   z_dens
            lensKernel  =   z_dens.dot(lensK)
            self.lensKernel=lensKernel
        return lensKernel

    def lensing_kernel_infty(self):
        """Mapping from an average delta in a lens redshfit bin
        to an average kappa in a source redshift at z=+infty
        """
        lensKernel =np.zeros((self.nz,self.nzl))
        kl      =   self.cosmo.angular_diameter_distance_z1z2(0.,self.zlcgrid).value*four_pi_G_over_c_squared()
        # Surface masss density in lens bin
        rhoM_ave=   self.cosmo.rho_m(self.zlcgrid)
        DaBin   =   self.cosmo.angular_diameter_distance_z1z2(self.zlbound[:-1],self.zlbound[1:]).value
        lensKernel= kl*rhoM_ave*DaBin
        return lensKernel
