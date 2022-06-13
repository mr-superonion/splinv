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
from .maputil import GausAtom
from astropy.wcs import WCS
import astropy.io.fits as pyfits
from astropy.cosmology import FlatLambdaCDM as Cosmo
import matplotlib.pyplot as plt
import matplotlib as mpl

class Cartesian():
    """
    pixlize in a TAN(Gnomonic)-prjected Cartesian Grid
    Rotation and coordinate distortion are not implemented
    Args:
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
        self.unit=unit

        # Line-of-sight direction for background galaxies
        if parser.has_section('sources'):
            if parser.has_option('sources','zbound'):
                zbound=np.array(json.loads(parser.get('sources','zbound')))
                nz  =   len(zbound)-1
            else:
                nz  =   parser.getint('sources','nz')
                assert nz>=1
                zmin=   parser.getfloat('sources','zmin')
                deltaz= parser.getfloat('sources','zscale')
                zmax=   zmin+deltaz*(nz+0.1)
                zbound= np.arange(zmin,zmax,deltaz)
            zcgrid  =   (zbound[:-1]+zbound[1:])/2.
        else:
            nz      =   1
            zbound  =   np.array([0.,100.])
            zcgrid  =   np.array([])
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
            self.__init_tan_plane(nx,ny,xmin,ymin)

        ## Gaussian smoothing in the transverse plane
        if parser.has_option('transPlane','smooth_scale'):
            self.sigma  =   parser.getfloat('transPlane','smooth_scale')*self.ratio
        else:
            self.sigma  =   -1
        self.sigma_pix  =   self.sigma/self.scale

        if parser.has_section('lens'):
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
            omega_m =   Default_OmegaM
        self.cosmo=Cosmo(H0=Default_h0*100.,Om0=omega_m)
        self.lensKernel =   None
        return

    def __init_wcs(self):
        """initialize a wcs (world coordinate system) for the projection
        """
        # prepare the WCS for 2D map
        wcs_header = {
            'CTYPE1': 'RA---TAN',
            'CUNIT1': 'deg',
            'CDELT1': self.scale,
            'CRPIX1': self.nx//2+1, # fits starting from 1 (but ndarray stars from 0)
            'CRVAL1': self.xcgrid[self.nx//2],
            'NAXIS1': self.nx,
            'CTYPE2': 'DEC--TAN',
            'CUNIT2': 'deg',
            'CDELT2': self.scale,
            'CRPIX2': self.ny//2+1, # fits starting from 1 (but ndarray stars from 0)
            'CRVAL2': self.ycgrid[self.ny//2],
            'NAXIS2': self.ny
        }
        self.wcs = WCS(wcs_header)
        return

    def reset_smooth_scale(self,sigma):
        """resets the smoothing scale
        (input the sigma in the same unit as self.unit)
        """
        self.sigma  =   sigma*self.ratio
        self.sigma_pix  =   self.sigma/self.scale
        return

    def __init_tan_plane(self,nx,ny,xmin,ymin):
        """setup Tan projection plane from basic parameters
        (no rotation,no flipping)
        Args:
            nx (int):       number of x bins [int]
            ny (int):       number of y bins [int]
            xmin (float):   minimum of x [deg]
            ymin (float):   minimum of y [deg]
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
        self.__init_wcs()
        return

    def setupTanPlane(self,ra=None,dec=None,pad=0.1,header=None):
        """setup Tan projection plane from (ra,dec) array or header
        (no rotation,no flipping)
        Args:
            ra (ndarray):   array of ra to project  [deg]
            dec (ndarray):  array of dec to project [deg]
            pad (float):    padding angular distance [deg]
            header (dict):  header with projection information
        Returns:
            ra (ndarray):   ra in the projected plane
            dec (ndarray):  dec in the projected plane
        """
        if (ra is not None) and (dec is not None):
            if header is not None:
                raise ValueError('Confusion: use catalog or header to setup grids?')
            # first try to define the center point of the plane
            self.ra0    =   (np.max(ra)+np.min(ra))/2.
            self.dec0   =   (np.max(dec)+np.min(dec))/2.
            self.cosdec0=   np.cos(self.dec0/180.*np.pi)
            self.sindec0=   np.sin(self.dec0/180.*np.pi)
            x,y         =   self.project_tan(ra,dec)
            xt          =   (np.max(x)+np.min(x))/2.
            yt          =   (np.max(y)+np.min(y))/2.

            # second try to define the center point of the plane
            self.ra0,self.dec0=self.iproject_tan(xt,yt)
            self.cosdec0=   np.cos(self.dec0/180.*np.pi)
            self.sindec0=   np.sin(self.dec0/180.*np.pi)
            x,y         =   self.project_tan(ra,dec)
            xt          =   (np.max(x)+np.min(x))/2.
            yt          =   (np.max(y)+np.min(y))/2.

            dxmin   =   self.ra0-(np.min(x)-pad)
            dxmax   =   (np.max(x)+pad)-self.ra0
            dnx     =   int(max(dxmin,dxmax)/self.scale+1.)
            dymin   =   self.dec0-(np.min(y)-pad)
            dymax   =   (np.max(y)+pad)-self.dec0
            dny     =   int(max(dymin,dymax)/self.scale+1.)

            # make sure we have even number of pixels in x and y
            self.nx =   2*dnx
            self.ny =   2*dny
            outcome =   (x,y)
        elif header is not None:
            assert abs(self.scale-header['CDELT1'])/self.scale<1e-2,\
                    'pixel scale inconsistent between ini file and header'
            assert abs(self.scale-header['CDELT2'])/self.scale<1e-2,\
                    'pixel scale inconsistent between ini file and header'
            self.ra0    =   header['CRVAL1']
            self.dec0   =   header['CRVAL2']
            self.cosdec0=   np.cos(self.dec0/180.*np.pi)
            self.sindec0=   np.sin(self.dec0/180.*np.pi)
            self.nx     =   int(header['NAXIS1'])
            self.ny     =   int(header['NAXIS2'])
            dnx         =   self.nx//2
            dny         =   self.ny//2
            outcome     =   None
        else:
            raise ValueError('user should input (ra,dec) or header')

        xmin    =   self.ra0-self.scale*(dnx+0.5)
        xmax    =   xmin+self.scale*(self.nx+0.1)
        ymin    =   self.dec0-self.scale*(dny+0.5)
        ymax    =   ymin+self.scale*(self.ny+0.1)

        # for ra direction
        # boundarys of the binning
        self.xbound= np.arange(xmin,xmax,self.scale)
        # center
        self.xcgrid= (self.xbound[:-1]+self.xbound[1:])/2.
        # same for dec direction
        self.ybound= np.arange(ymin,ymax,self.scale)
        self.ycgrid= (self.ybound[:-1]+self.ybound[1:])/2.
        self.shape = (self.nz,self.ny,self.nx)
        self.__init_wcs()
        return outcome

    def project_tan(self,ra,dec,pix=False):
        """TAN(Gnomonic)-prjection of sky coordiantes
        (no rotation,no flipping)
        Args:
            ra (ndarray):   sky coordinate---ra [deg]
            dec (ndarray):  sky coordinate---dec [deg]
            pix (bool):     return in unit of pixel or not
        Returns:
            x1 (ndarray):   pixel coordinate x1 [deg]
            x2 (ndarray):   pixel coordinate x2 [deg]
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
        Args:
            x1 (ndarray):   x1 pixel coord [deg]
            x2 (ndarray):   x2 pixel coord [deg]
            pix (bool):     input in unit of pixel or not (default=False)
        Returns:
            ra (ndarray):   sky coordinate [deg]
            dec (ndarray):  sky coordinate [deg]
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

    def pixelize_data(self,x,y,z,v=None,ws=None,smooth_method='pix',return_mask=False):
        """pixelize catalog into the cartesian grid
        Args:
            x (ndarray):    ra of sources [deg]
            y (ndarray):    dec of sources [deg]
            z (ndarray):    redshifts of sources
            v (ndarray):    measurements. (e.g., shear, kappa, default: None)
            ws (ndarray):   galaxy weights (default: None).
            smooth_method (str):
                            method to convolve with the Gaussian kernel
                            ('gal' or 'pix' default: 'pix')
            return_mask (bool):
                            return mask or not (default: False)
        Returns:
            out (ndarray):  pixelized data
                            when v==None, returns the galaxy numbers in pixels
        """
        if z is None:
            # This is for 2D pixeliztion
            assert self.shape[0]==1
            z = np.ones(len(x))
        if ws is None:
            # Without weight
            ws = np.ones(len(x))
        if smooth_method=='gal':
            # conduct galaxy level smooth: smooth then pixelize (slow)
            if self.sigma>0.:
                return self._pixelize_data_gal(x,y,z,v,ws)
            else:
                raise ValueError("when method is 'gal', smooth_scale must >0")
        elif smooth_method=='pix':
            # conduct pixel level smooth: pixelize then smooth (fast)
            return self._pixelize_data_pix(x,y,z,v,ws,return_mask)
        else:
            raise ValueError("method must be 'pix' or 'gal'")

    def _pixelize_data_pix(self,x,y,z,v=None,ws=np.empty(0),return_mask=False):
        """pixelize data and then do smoothing on galaxy level (fast)
        """
        if v is not None:
            # pixelize value field
            dataOut =   np.histogramdd((z,y,x),bins=(self.zbound,self.ybound,self.xbound),weights=v*ws)[0]
        else:
            dataOut =   []
        # pixelize weight field
        weightOut=  np.histogramdd((z,y,x),bins=(self.zbound,self.ybound,self.xbound),weights=ws)[0]
        if self.sigma_pix>0.01: #0.01 arcmin
            # Gaussian Kernel in Fourier space
            # (normalize to flux-1)
            gausKer =   GausAtom(ny=self.ny,nx=self.nx,sigma=self.sigma_pix,fou=True)
            norm    =   gausKer[0,0]
            gausKer /=  norm
            # smothing with Gausian Kernel
            # (for both weight and value)
            weightOut=  np.fft.ifft2(np.fft.fft2(weightOut)*gausKer).real
            if v is not None:
                dataOut=np.fft.ifft2(np.fft.fft2(dataOut)*gausKer).real

        # intial guess of the mask to estmiate threshold
        mask            =   weightOut>1e-3
        thres           =   np.mean(weightOut[mask])/6.
        # final mask
        mask            =   weightOut>thres
        weightOut[~mask]=   0 # pixels with values lower than threshold are set to 0
        if v is not None:
            # avoid weight is zero
            dataOut[mask]   =   dataOut[mask]/weightOut[mask]
            dataOut[~mask]  =   0.
            if not return_mask:
                return dataOut
            else:
                return dataOut,mask
        else:
            if not return_mask:
                return weightOut
            else:
                return weightOut,mask

    def _pixelize_data_gal(self,x,y,z,v=None,ws=np.empty(0)):
        """do smoothing on galaxy level and then pixelize data (slow)
        """
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
        Args:
            poz_grids (ndarray):poz's bin; if None, do not include any photoz uncertainty
                                (default=None)
            poz_best (ndarray): galaxy's best photoz measurements, used for galaxy binning
            poz_data (ndarray): galaxy's POZ measurements, used for deriving lensing kernel
                                shape=(len(galaxy), len(poz_grids))
            z_dens (ndarray):   average POZ in source bins
            deltaIn (bool):     mapping from delta to kappa (True) or from mass to kappa (False)
                                (default=True)
        Returns:
            out (ndarray):      lensing kernel
        """

        assert (poz_data is not None)==(poz_best is not None), \
            'Please provide both photo-z bins and photo-z data'
        assert (self.nzl==1)==(self.nz==1), \
            'number of lens plane and source plane'
        if self.nzl<=1:
            return np.ones((self.nz,self.nzl))
        out =   np.zeros((self.nz,self.nzl))
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
                    out[:,i]=kl*rhoM_ave*DaBin
                else:
                    # When the input is M_200/1e14
                    out[:,i]=kl*1e14
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
                if poz_data is None:
                    raise ValueError('Cannot estimate n(z) if poz_data is None')
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
            out  =   z_dens.dot(lensK)
            self.lensKernel=out
        return out

    def lensing_kernel_infty(self):
        """Mapping from an average delta in a lens redshfit bin
        to an average kappa in a source redshift at z=+infty
        Returns:
            lensKernel (ndarray):     lensing kernel
        """
        lensKernel =np.zeros((self.nz,self.nzl))
        kl      =   self.cosmo.angular_diameter_distance_z1z2(0.,self.zlcgrid).value*four_pi_G_over_c_squared()
        # Surface masss density in lens bin
        rhoM_ave=   self.cosmo.critical_density(self.zlcgrid).to_value(unit=rho_unt)*self.cosmo.Om(self.zlcgrid)
        DaBin   =   self.cosmo.angular_diameter_distance_z1z2(self.zlbound[:-1],self.zlbound[1:]).value
        lensKernel= kl*rhoM_ave*DaBin
        return lensKernel

    def make_plot2D(self,mapIn,mask=None,title='',histrange=None):
        """Plot the 2D pixelized map
        Args:
            mapIn (ndarray):    input 2D map
            mask (ndarray):     input 2D mask
            title (str):        title of the plot, e.g., 'HSC: g1', 'B-mode'
            histrange (tuple):  range of the histogram
        Returns:
            fig1 (figure):      matplotlib figure for 2D map
            fig2 (figure):      matplotlib figure for pixel histogram
        """
        dim=len(mapIn.shape)
        if dim!=2:
            raise ValueError('Input map should be 2D, but get a %dD array' %dim)
        pratio=int(mapIn.shape[1]/mapIn.shape[0]+0.5)
        # pixel value list
        if mask is not None:
            assert mask.shape==mapIn.shape, 'The input map and mask have different shape'
            tt=np.ravel(np.ravel(mapIn))[np.ravel(mask)]
        else:
            tt=np.ravel(np.ravel(mapIn))
        vmin,vmax=np.percentile(tt,q=[0.02,99.98])
        # determine the color bar
        if vmin*vmax>=0:
            cmap='RdYlBu_r'
            gcolor='white'
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        else:
            cmap='seismic'
            gcolor='black'
            if np.abs(vmin+vmax)<max(-vmin,vmax)*0.3:
                vmax2=(vmax-vmin)/2.
                vmin2=(vmax-vmin)/-2.
                vmax=vmax2*1.25;vmin=vmin2*1.25
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        # plot for 2D map
        interpolate=None
        fig1=plt.figure(figsize=(4*pratio,4))
        ax=fig1.add_subplot(1,1,1,projection=self.wcs,aspect='equal')
        ax.set_title(title,fontsize=20)
        imap=ax.imshow(mapIn,origin='lower',norm=norm,cmap=cmap,interpolation=interpolate)
        ax.grid(color=gcolor, ls='dotted')
        ax.coords[0].set_major_formatter('d.d')
        ax.set_xlabel('ra [deg]',fontsize=20)
        ax.set_ylabel('dec [deg]',fontsize=20)
        fig1.colorbar(imap)

        # plot for pixle histogram
        fig2=plt.figure(figsize=(7,6))
        ax2=fig2.add_subplot(1,1,1)
        if histrange is not None:
            ax2.hist(tt,bins=20,range=histrange,histtype='step',density=True)
        else:
            ax2.hist(tt,bins=20,histtype='step',density=True)
        ax2.set_yscale('log')
        if vmin*vmax<0.:
            ax2.set_xlim(vmin*1.6,vmax*1.6)
        else:
            ax2.set_xlim(vmin,vmax)
        ax2.set_ylabel('PDF')
        ax2.set_xlabel(title.split(':')[-1])
        ax2.set_title(title,fontsize=20)
        ax2.grid()
        fig2.tight_layout()
        return fig1,fig2

    def write_fits(self,fname,data,with_wcs=False,overwrite=False):
        """wrapper of pyfits.writeto. Write map to disk
        Args:
            fname (str):        file name
            data (ndarray):     map to write to disk
            with_wcs (bool):    whether write wcs header (default: False)
            overwrite (bool):   whether overwrite existing file
        """
        if with_wcs:
            pyfits.writeto(fname,data,header=self.wcs.to_header(),overwrite=overwrite)
        else:
            pyfits.writeto(fname,data,overwrite=overwrite)
        return

    def read_fits(self,fname,with_wcs=False):
        """wrapper of pyfits.read
        Args:
            fname (str):        file name
            with_wcs (bool):    whether read wcs header (default: False)
        Returns:
            data (ndarray):     map
        """
        data=pyfits.getdata(fname)
        if with_wcs:
            wcs_header=pyfits.getheader(fname)
            self.setupTanPlane(header=wcs_header)
        return data
