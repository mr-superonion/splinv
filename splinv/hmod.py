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
import numpy as np
from .default import *
from .grid import Cartesian
from .maputil import TophatAtom
import scipy.special as spfun
import astropy.io.fits as pyfits
from astropy.cosmology import FlatLambdaCDM as Cosmo

def zMeanBin(zMin,dz,nz):
    return np.arange(zMin,zMin+dz*nz,dz)+dz/2.

def haloCS02SigmaAtom(r_s,ny,nx=None,c=9.,sigma_pix=None,fou=True,lnorm=2.):
    """
    Make haloTJ03 halo (normalized) from Fourier space following
    Eq. 81 and 82, Cooray & Sheth (2002, Physics Reports, 372,1):
    https://arxiv.org/pdf/astro-ph/0206508.pdf

    Parameters:
        r_s:    scale radius (iuo pixel) [float]
        ny,nx:  number of pixel in y and x directions [int]
        c:      truncation ratio (concentration) [float]
        sigma_pix: sigma for Gaussian smoothing (iuo pixel) [float]
        fou:    in Fourier space [bool]
        lnorm:  l-norm for normalization

    """
    if nx is None:
        nx=ny
    x,y =   np.meshgrid(np.fft.fftfreq(nx),np.fft.fftfreq(ny))
    x   *=  (2*np.pi);y*=(2*np.pi)
    rT  =   np.sqrt(x**2+y**2)
    if r_s<=0.1:
        # point mass in Fourier space
        atom=np.ones((ny,nx))
    else:
        # NFW halo in Fourier space
        A       =   1./(np.log(1+c)-c/(1.+c))
        r       =   rT*r_s
        mask    =   r>0.001
        atom    =   np.zeros_like(r,dtype=float)
        r1      =   r[mask]
        si1,ci1 =   spfun.sici((1+c)*r1)
        si2,ci2 =   spfun.sici(r1)
        atom[mask]= A*(np.sin(r1)*(si1-si2)-np.sin(c*r1)/(1+c)/r1+np.cos(r1)*(ci1-ci2))
        r0      =   r[~mask]
        atom[~mask]=1.+A*(c+c**3/(6*(1 + c))+1/4.*(-2.*c-c**2.-2*np.log(1+c)))*r0**2.

    if sigma_pix is not None:
        if sigma_pix>0.1:
            # Gaussian smoothing
            atom    =   atom*np.exp(-(rT*sigma_pix)**2./2.)
        else:
            # top-hat smoothing
            atom    =   atom*TophatAtom(width=1.,ny=ny,nx=nx,fou=True)

    if fou:
        # Fourier space
        if lnorm>0.:
            norm=   (np.sum(atom**lnorm)/(nx*ny))**(1./lnorm)
        else:
            norm=   1.
    else:
        # configuration space
        atom    =   np.real(np.fft.ifft2(atom))
        if lnorm>0.:
            norm=   (np.sum(atom**lnorm))**(1./lnorm)
        else:
            norm=   1.
    return atom/norm

def mc2rs(mass,conc,redshift,omega_m=Default_OmegaM):
    """
    Get the scale radius of NFW halo from mass and redshift
    Parameters:
        mass:       Mass defined using a spherical overdensity of 200 times the
                    critical density of the universe, in units of M_solar/h.
        conc:       Concentration parameter, i.e., ratio of virial radius to NFW
                    scale radius.
        redshift:   Redshift of the halo.
    Returns:
        scale radius in arcsec
    """
    cosmo   =   Cosmo(H0=Default_h0*100.,Om0=omega_m)
    z       =   redshift
    #a       =   1./(1.+z)
    # angular distance in Mpc/h
    DaLens  =   cosmo.angular_diameter_distance_z1z2(0.,z).value
    # E(z)^{-1}
    ezInv   =   cosmo.inv_efunc(z)
    # critical density (in unit of M_sun h^2 / Mpc^3)
    #rho_cZ  =   cosmo.critical_density(self.z).to_value(unit=rho_unt)
    rvir    =   1.63e-5*(mass*ezInv**2)**(1./3.) # in Mpc/h
    rs      =   rvir/conc
    #A       =   1./(np.log(1+conc)-(conc)/(1+conc))
    #delta_nfw   =   200./3*conc**3*A
    # convert to angular radius in unit of arcsec
    scale       =   rs / DaLens
    arcmin2rad  =   np.pi/180./60.
    rs_arcmin   =   scale/arcmin2rad
    return rs_arcmin

class ksmap():
    """
    A Class for 2D Kaiser-Squares transform:
    Kaiser & Squires (1993, ApJ, 404, 2)
    https://articles.adsabs.harvard.edu/pdf/1993ApJ...404..441K

    Parameters:
    ny,nx: number of pixels in y and x directions

    Methods:
    itransform:

    transform:
    """
    def __init__(self,ny,nx):
        self.shape   =   (ny,nx)
        self.e2phiF  =   self.e2phiFou()

    def e2phiFou(self):
        ny,nx   =   self.shape
        e2phiF  =   np.zeros(self.shape,dtype=np.complex128)
        for j in range(ny):
            jy  =   (j+ny//2.)%ny-ny//2.
            jy  =   np.float64(jy/ny)
            for i in range(nx):
                ix  =   (i+nx//2.)%nx-nx//2.
                ix  =   np.float64(ix/nx)
                if i==0 and j==0:
                    e2phiF[j,i] =   0.
                else:
                    r2  =   ix**2.+jy**2.
                    e2phiF[j,i] =   (ix**2.-jy**2.)/r2+(2j*ix*jy/r2)
        return e2phiF*np.pi

    def itransform(self,gMap,inFou=True,outFou=True):
        """
        K-S Transform from gamma map to kappa map

        Parameters:
        kMap:   input gamma map
        inFou:  input in Fourier space? [default:True=yes]
        outFou: output in Fourier space? [default:True=yes]
        """
        assert gMap.shape[-2:]==self.shape
        if not inFou:
            gMap =   np.fft.fft2(gMap)
        kOMap    =   gMap*np.conjugate(self.e2phiF*np.pi)
        if not outFou:
            kOMap    =   np.fft.ifft2(kOMap)
        return kOMap

    def transform(self,kMap,inFou=True,outFou=True):
        """
        K-S Transform from kappa map to gamma map

        Parameters:
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

class nfwHalo(Cosmo):
    """
    Parameters:
        mass:       Mass defined using a spherical overdensity of 200 times the
                    critical density of the universe, in units of M_solar/h.
        conc:       Halo concentration, virial radius / scale radius.
        redshift:   Redshift of the halo.
        ra:         ra of halo center  [arcsec].
        dec:        dec of halo center [arcsec].
        omega_m:    Omega_matter to pass to Cosmology constructor, omega_l is
                    set to 1-omega_matter. (default: Default_OmegaM)
    """
    def __init__(self,ra,dec,redshift,mass,conc=None,rs=None,omega_m=Default_OmegaM):
        # Redshift and Geometry
        ## ra dec
        self.ra     =   ra
        self.dec    =   dec
        Cosmo.__init__(self,H0=Default_h0*100.,Om0=omega_m)
        self.z      =   float(redshift)
        self.a      =   1./(1.+self.z)
        # angular distance in Mpc/h
        self.DaLens =   self.angular_diameter_distance_z1z2(0.,self.z).value
        # critical density
        # in unit of M_solar / Mpc^3
        rho_cZ      =   self.critical_density(self.z).to_value(unit=rho_unt)

        self.M      =   float(mass)
        # First, we get the virial radius, which is defined for some spherical
        # overdensity as 3 M / [4 pi (r_vir)^3] = overdensity. Here we have
        # overdensity = 200 * rhocrit, to determine r_vir (angular distance).
        # The factor of 1.63e-5 [h^(-2/3.)] comes from the following set of prefactors:
        # (3 / (4 pi * 200 * rhocrit))^(1/3), where rhocrit = 2.8e11 h^2
        # M_solar / Mpc^3.
        # (DH=C_LIGHT/1e3/100/h,rho_crit=1.5/four_pi_G_over_c_squared()/(DH)**2.)
        ezInv       =   self.inv_efunc(redshift)
        self.rvir   =   1.63e-5*(self.M*ezInv**2)**(1./3.) # in Mpc/h
        if conc is not None:
            self.c  =   float(conc)
            # scale radius
            self.rs =   self.rvir/self.c
            if rs is not None:
                assert abs(self.rs-rs)<0.01, 'input rs is different from derived'
        elif rs is not None:
            self.rs =   float(rs)
            self.c  =  self.rvir/self.rs
        else:
            raise ValueError("need to give conc or rs, at least one")

        self.A      =   1./(np.log(1+self.c)-(self.c)/(1+self.c))
        self.delta_nfw =200./3*self.c**3*self.A
        # convert to angular radius in unit of arcsec
        scale       =   self.rs / self.DaLens
        arcsec2rad  =   np.pi/180./3600.
        self.rs_arcsec =scale/arcsec2rad

        # Second, derive the charateristic matter density
        # within virial radius at redshift z
        self.rho_s=   rho_cZ*self.delta_nfw
        return

    def DdRs(self,ra_s,dec_s):
        """Calculate 'x' the radius r in units of the NFW scale
        radius, r_s.
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        x = ((ra_s - self.ra)**2 + (dec_s - self.dec)**2)**0.5/self.rs_arcsec
        return x

    def sin2phi(self,ra_s,dec_s):
        """Calculate cos2phi and sin2phi with reference to the halo center
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # pure tangential shear, no cross component
        dx = ra_s - self.ra
        dy = dec_s - self.dec
        drsq = dx*dx+dy*dy
        return np.divide(2*dx*dy, drsq, where=(drsq != 0.))

    def cos2phi(self,ra_s,dec_s):
        """Calculate cos2phi and sin2phi with reference to the halo center
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # pure tangential shear, no cross component
        dx = ra_s - self.ra
        dy = dec_s - self.dec
        drsq = dx*dx+dy*dy
        return np.divide(dx*dx-dy*dy, drsq, where=(drsq != 0.))

    def lensKernel(self,z_s):
        """Lensing kernel from surface density at lens redshfit to source redshift
        to kappa at source redshift
        Parameters:
            z_s:        redshift of sources.
        """
        # convenience: call with single number
        if not isinstance(z_s, np.ndarray):
            return self.lensKernel(np.array([z_s], dtype='float'))[0]
        # lensing weights: the only thing that depends on z_s
        # First mask the data with z<z_l
        k_s =   np.zeros(len(z_s))
        mask=   z_s>self.z
        k_s[mask] =   self.angular_diameter_distance_z1z2(self.z,z_s[mask])*self.DaLens\
                /self.angular_diameter_distance_z1z2(0.,z_s[mask])*four_pi_G_over_c_squared()
        return k_s


class nfwWB00(nfwHalo):
    """
    Integral functions of an untruncated spherical NFW profile:
    Eq. 11, Wright & Brainerd (2000, ApJ, 534, 34) --- Surface Density
    and Eq. 13 14 15 --- Excess Surface Density
    Parameters:

        mass:       Mass defined using a spherical overdensity of 200 times the
                    critical density of the universe, in units of M_solar/h.
        conc:       Concentration parameter virial radius / scale radius.
        redshift:   Redshift of the halo.
        ra:         ra of halo center  [arcsec].
        dec:        dec of halo center [arcsec].
        omega_m:    Omega_matter to pass to Cosmology constructor, omega_l is
                    set to 1-omega_matter. (default: Default_OmegaM)
    """
    def __init__(self,ra,dec,redshift,mass=None,conc=None,rs=None,omega_m=Default_OmegaM):
        nfwHalo.__init__(self,ra,dec,redshift,mass=mass,conc=conc,rs=rs,omega_m=omega_m)

    def __Sigma(self,x):
        out = np.zeros_like(x, dtype=float)

        # 3 cases: x < 1, x > 1, and |x-1| < 0.001
        mask = np.where(x < 0.999)[0]
        a = ((1 - x[mask])/(x[mask] + 1))**0.5
        out[mask] = 2/(x[mask]**2 - 1) * (1 - 2*np.arctanh(a)/(1-x[mask]**2)**0.5)

        mask = np.where(x > 1.001)[0]
        a = ((x[mask] - 1)/(x[mask] + 1))**0.5
        out[mask] = 2/(x[mask]**2 - 1) * (1 - 2*np.arctan(a)/(x[mask]**2- 1)**0.5)

        # the approximation below has a maximum fractional error of 7.4e-7
        mask = np.where((x >= 0.999) & (x <= 1.001))[0]
        out[mask] = (22./15. - 0.8*x[mask])
        return out* self.rs * self.rho_s

    def Sigma(self,ra_s,dec_s):
        """Calculate Surface Density (Sigma) of halo.
        Equation (11) in Wright & Brainerd (2000, ApJ, 534, 34).
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # convenience: call with single number
        assert isinstance(ra_s,np.ndarray)==isinstance(dec_s,np.ndarray),\
            'ra_s and dec_s do not have same type'
        if not isinstance(ra_s,np.ndarray):
            ra_sA=np.array([ra_s], dtype='float')
            dec_sA=np.array([dec_s], dtype='float')
            return self.Sigma(ra_sA,dec_sA)[0]
        assert len(ra_s)==len(dec_s),\
            'input ra and dec have different length '
        x=self.DdRs(ra_s,dec_s)
        return self.__Sigma(x)

    def __DeltaSigma(self,x):
        out = np.zeros_like(x, dtype=float)
        """
        # 4 cases:
        # x > 1,0.01< x < 1,|x-1| < 0.001
        # x<0.01
        """
        mask = np.where(x > 1.001)[0]
        a = ((x[mask]-1.)/(x[mask]+1.))**0.5
        out[mask] = x[mask]**(-2)*(4.*np.log(x[mask]/2)+8.*np.arctan(a)\
                /(x[mask]**2 - 1)**0.5)*self.rs * self.rho_s-self.__Sigma(x[mask])
        # Equivalent but usually faster than mask = (x < 0.999)
        mask = np.where((x < 0.999) & (x> 0.01))[0]
        a = ((1.-x[mask])/(x[mask]+1.))**0.5
        out[mask] = x[mask]**(-2)*(4.*np.log(x[mask]/2)+8.*np.arctanh(a)\
                /(1-x[mask]**2)**0.5)*self.rs * self.rho_s-self.__Sigma(x[mask])
        """
        # the approximation below has a maximum fractional error of 2.3e-7
        """
        mask = np.where((x >= 0.999) & (x <= 1.001))[0]
        out[mask] = (4.*np.log(x[mask]/2)+40./6. - 8.*x[mask]/3.)*self.rs * self.rho_s\
                    -self.__Sigma(x[mask])
        """
        # the approximation below has a maximum fractional error of 1.1e-7
        """
        mask        =   np.where(x <= 0.01)[0]
        out[mask]   =   4.*(0.25 + 0.125 * x[mask]**2 * \
                (3.25 + 3.0*np.log(x[mask]/2)))*self.rs * self.rho_s
        return out

    def DeltaSigma(self,ra_s,dec_s):
        """Calculate excess surface density of halo.
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # convenience: call with single number
        assert isinstance(ra_s,np.ndarray)==isinstance(dec_s,np.ndarray),\
            'ra_s and dec_s do not have same type'
        if not isinstance(ra_s,np.ndarray):
            ra_sA=np.array([ra_s], dtype='float')
            dec_sA=np.array([dec_s], dtype='float')
            return self.DeltaSigma(ra_sA,dec_sA)[0]
        assert len(ra_s)==len(dec_s),\
            'input ra and dec have different length '
        x=self.DdRs(ra_s,dec_s)
        return self.__DeltaSigma(x)

    def DeltaSigmaComplex(self,ra_s,dec_s):
        """Calculate excess surface density of halo.
        return a complex array Delta Sigma_1+ i Delta Sigma_2
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # convenience: call with single number
        assert isinstance(ra_s,np.ndarray)==isinstance(dec_s,np.ndarray),\
            'ra_s and dec_s do not have same type'
        if not isinstance(ra_s,np.ndarray):
            ra_sA=np.array([ra_s], dtype='float')
            dec_sA=np.array([dec_s], dtype='float')
            return self.DeltaSigmaComplex(ra_sA,dec_sA)[0]
        assert len(ra_s)==len(dec_s),\
            'input ra and dec have different length '
        x=self.DdRs(ra_s,dec_s)
        DeltaSigma=self.__DeltaSigma(x)
        DeltaSigma1 = -DeltaSigma*self.cos2phi(ra_s,dec_s)
        DeltaSigma2 = -DeltaSigma*self.sin2phi(ra_s,dec_s)
        return DeltaSigma1+1j*DeltaSigma2

    def SigmaAtom(self,pix_scale,ngrid,xc=None,yc=None):
        """NFW Atom on Grid
        Parameters:
            pix_scale:    pixel sacle [arcsec]
            ngrid:        number of pixels on x and y axis
        """
        if xc is None:
            xc  =   self.ra
        if yc is None:
            yc  =   self.dec

        X   =   (np.arange(ngrid)-ngrid/2.)*pix_scale+xc
        Y   =   (np.arange(ngrid)-ngrid/2.)*pix_scale+yc
        x,y =   np.meshgrid(X,Y)
        atomReal=self.Sigma(x.ravel(),y.ravel()).reshape((ngrid,ngrid))
        return atomReal

    def DeltaSigmaAtom(self,pix_scale,ngrid,xc=None,yc=None):
        """NFW Atom on Grid
        Parameters:
            pix_scale:    pixel sacle [arcsec]
            ngrid:        number of pixels on x and y axis
        """
        if xc is None:
            xc  =   self.ra
        if yc is None:
            yc  =   self.dec

        X   =   (np.arange(ngrid)-ngrid/2.)*pix_scale+xc
        Y   =   (np.arange(ngrid)-ngrid/2.)*pix_scale+yc
        x,y =   np.meshgrid(X,Y)
        atomReal=self.DeltaSigma(x.ravel(),y.ravel()).reshape((ngrid,ngrid))
        return atomReal

class nfwTJ03(nfwHalo):
    """
    Integral functions of an truncated spherical NFW profile:
    Eq.27, Takada & Jain (2003, MNRAS, 340, 580) --- Surface Density,
    and Eq.17, Takada & Jain (2003, MNRAS, 344, 857) --- Excess Surface Density
    Parameters:
        mass:       Mass defined using a spherical overdensity of 200 times the
                    critical density of the universe, in units of M_solar/h.
        conc:       Concentration parameter virial radius / scale radius.
        redshift:   Redshift of the halo.
        ra:         ra of halo center  [arcsec].
        dec:        dec of halo center [arcsec].
        omega_m:    Omega_matter to pass to Cosmology constructor, omega_l is
                    set to 1-omega_matter. (default: Default_OmegaM)
    """
    def __init__(self,ra,dec,redshift,mass=None,conc=None,rs=None,omega_m=Default_OmegaM):
        nfwHalo.__init__(self,ra,dec,redshift,mass=mass,conc=conc,rs=rs,omega_m=omega_m)

    def __Sigma(self,x0):
        c   = float(self.c)
        out = np.zeros_like(x0, dtype=float)

        # 3 cases: x < 1-0.001, x > 1+0.001, and |x-1| < 0.001
        mask = np.where(x0 < 0.999)[0]
        x=x0[mask]
        out[mask] = -np.sqrt(c**2.-x**2.)/(1-x**2.)/(1+c)+\
            1./(1-x**2.)**1.5*np.arccosh((x**2.+c)/x/(1.+c))

        mask = np.where((x0 > 1.001) & (x0<c))[0]
        x=x0[mask]
        out[mask] = -np.sqrt(c**2.-x**2.)/(1-x**2.)/(1+c)-\
            1./(x**2.-1)**1.5*np.arccos((x**2.+c)/x/(1.+c))

        mask = np.where((x0 >= 0.999) & (x0 <= 1.001))[0]
        x=x0[mask]
        out[mask] = (-2.+c+c**2.)/(3.*np.sqrt(-1.+c)*(1+c)**(3./2))\
            +((2.-c-4.*c**2.-2.*c**3.)*(x-1.))/(5.*np.sqrt(-1.+c)*(1+c)**(5/2.))

        mask = np.where(x0 >= c)[0]
        out[mask]=0.
        return out* self.rs * self.rho_s*2.

    def Sigma(self,ra_s,dec_s):
        """Calculate Surface Density (Sigma) of halo.
        Takada & Jain(2003, MNRAS, 340, 580) Eq.27
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """

        # convenience: call with single number
        assert isinstance(ra_s,np.ndarray)==isinstance(dec_s,np.ndarray),\
            'ra_s and dec_s do not have same type'
        if not isinstance(ra_s,np.ndarray):
            ra_sA=np.array([ra_s], dtype='float')
            dec_sA=np.array([dec_s], dtype='float')
            return self.Sigma(ra_sA,dec_sA)[0]
        assert len(ra_s)==len(dec_s),\
            'input ra and dec have different length '
        x=self.DdRs(ra_s,dec_s)
        return self.__Sigma(x)

    def __DeltaSigma(self,x0):
        c   = float(self.c)
        out = np.zeros_like(x0, dtype=float)

        # 4 cases:
        # x < 1-0.001,|x-1| <= 0.001
        # 1.001<x<=c, x>c

        mask = np.where(x0 < 0.0001)[0]
        out[mask]=1./2.

        mask = np.where((x0 < 0.999) & (x0>0.0001) )[0]
        x=x0[mask]
        out[mask] = (-2.*c+((2.-x**2.)*np.sqrt(c**2.-x**2.))/(1-x**2))/((1+c)*x**2.)\
            +((2-3*x**2)*np.arccosh((c+x**2)/((1.+c)*x)))/(x**2*(1-x**2.)**1.5)\
            +(2*np.log(((1.+c)*x)/(c+np.sqrt(c**2-x**2))))/x**2

        mask = np.where((x0 > 1.001) & (x0< c))[0]
        x=x0[mask]
        out[mask] = (-2.*c+((2.-x**2.)*np.sqrt(c**2.-x**2.))/(1-x**2))/((1+c)*x**2.)\
            -((2-3*x**2)*np.arccos((c+x**2)/((1.+c)*x)))/(x**2*(-1+x**2.)**1.5)\
            +(2*np.log(((1.+c)*x)/(c+np.sqrt(c**2-x**2))))/x**2

        mask = np.where((x0 >= 0.999) & (x0 <= 1.001))[0]
        x=x0[mask]
        out[mask] = (10*np.sqrt(-1.+c**2)+c*(-6-6*c+11*np.sqrt(-1.+c**2))\
            +6*(1 + c)**2*np.log((1. + c)/(c +np.sqrt(-1.+c**2))))/(3.*(1+c)**2)-\
            (-1.+x)*((94 + c*(113 + 60*np.sqrt((-1.+c)/(1 + c))+4*c*(-22 + 30*np.sqrt((-1 + c)/(1 + c)) \
            + c*(-26 + 15*np.sqrt((-1 + c)/(1 + c))))))/(15.*(1.+c)**2*np.sqrt(-1.+c**2))- 4*np.log(1.+c)+\
            4*np.log(c +np.sqrt(-1.+c**2)))

        mask = np.where(x0 >= c)[0]
        x=x0[mask]
        out[mask] = 2./self.A/x**2.
        return out*self.rs * self.rho_s*2.

    def DeltaSigma(self,ra_s,dec_s):
        """Calculate excess surface density of halo according to
        Takada & Jain (2003, MNRAS, 344, 857) Eq.17 -- Excess Surface Density
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # convenience: call with single number
        assert isinstance(ra_s,np.ndarray)==isinstance(dec_s,np.ndarray),\
            'ra_s and dec_s do not have same type'
        if not isinstance(ra_s,np.ndarray):
            ra_sA=np.array([ra_s], dtype='float')
            dec_sA=np.array([dec_s], dtype='float')
            return self.DeltaSigma(ra_sA,dec_sA)[0]
        assert len(ra_s)==len(dec_s),\
            'input ra and dec have different length '
        x=self.DdRs(ra_s,dec_s)
        return self.__DeltaSigma(x)

    def DeltaSigmaComplex(self,ra_s,dec_s):
        """Calculate excess surface density of halo.
        return a complex array Delta Sigma_1+ i Delta Sigma_2
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # convenience: call with single number
        assert isinstance(ra_s,np.ndarray)==isinstance(dec_s,np.ndarray),\
            'ra_s and dec_s do not have same type'
        if not isinstance(ra_s,np.ndarray):
            ra_sA=np.array([ra_s], dtype='float')
            dec_sA=np.array([dec_s], dtype='float')
            return self.DeltaSigmaComplex(ra_sA,dec_sA)[0]
        assert len(ra_s)==len(dec_s),\
            'input ra and dec have different length '
        x=self.DdRs(ra_s,dec_s)
        DeltaSigma=self.__DeltaSigma(x)
        DeltaSigma1 = -DeltaSigma*self.cos2phi(ra_s,dec_s)
        DeltaSigma2 = -DeltaSigma*self.sin2phi(ra_s,dec_s)
        return DeltaSigma1+1j*DeltaSigma2

    def SigmaAtom(self,pix_scale,ngrid,xc=None,yc=None):
        """NFW Sigma on Grid
        Parameters:
            pix_scale:    pixel sacle [arcsec]
            ngrid:        number of pixels on x and y axis
        """
        if xc is None:
            xc  =   self.ra
        if yc is None:
            yc  =   self.dec

        X   =   (np.arange(ngrid)-ngrid/2.)*pix_scale+xc
        Y   =   (np.arange(ngrid)-ngrid/2.)*pix_scale+yc
        x,y =   np.meshgrid(X,Y)
        atomReal=self.Sigma(x.ravel(),y.ravel()).reshape((ngrid,ngrid))
        return atomReal

    def DeltaSigmaAtom(self,pix_scale,ngrid,xc=None,yc=None):
        """NFW Delta Sigma on Grid
        Parameters:
            pix_scale:    pixel sacle [arcsec]
            ngrid:        number of pixels on x and y axis
        """
        if xc is None:
            xc  =   self.ra
        if yc is None:
            yc  =   self.dec

        X   =   (np.arange(ngrid)-ngrid/2.)*pix_scale+xc
        Y   =   (np.arange(ngrid)-ngrid/2.)*pix_scale+yc
        x,y =   np.meshgrid(X,Y)
        atomReal=self.DeltaSigma(x.ravel(),y.ravel()).reshape((ngrid,ngrid))
        return atomReal

class nfwCS02_grid(Cartesian):
    """
    Integral functions of an truncated spherical NFW profile:
    Eq. 81 and 82, Cooray & Sheth (2002, Physics Reports, 372,1
    https://arxiv.org/pdf/astro-ph/0206508.pdf),
    for surface density and excess surface density
    Parameters:
        parser
    """
    def __init__(self,parser):
        Cartesian.__init__(self,parser)
        self.ks2D    =   ksmap(self.ny,self.nx)
        return

    def add_halo(self,halo):
        lk      =   halo.lensKernel(self.zcgrid)
        rpix    =   halo.rs_arcsec/self.scale/3600.

        sigma   =   haloCS02SigmaAtom(rpix,ny=self.ny,nx=self.nx,\
                    sigma_pix=self.sigma_pix,c=halo.c,fou=True)
        snorm   =   sigma[0,0]
        dr      =   halo.DaLens*self.scale/180*np.pi
        snorm   =   halo.M/dr**2./snorm
        sigma   =   sigma*snorm
        dsigma  =   np.fft.fftshift(self.ks2D.transform(sigma,\
                    inFou=True,outFou=False))
        sigma   =   np.fft.fftshift(np.fft.ifft2(sigma)).real
        shear   =   dsigma[None,:,:]*lk[:,None,None]
        kappa   =   sigma[None,:,:]*lk[:,None,None]
        return kappa,shear

class nfwShearlet2D():
    """
    A Class for 2D nfwlet transform
    with different angular scale in different redshift plane

    Parameters:
        nframe  :   number of frames
        ny,nx   :   size of the field (pixel)
        smooth_scale:   scale radius of Gaussian smoothing kernal (pixel)
        nlp     :   number of lens plane
        nz      :   number of source plane

    Methods:
        itransform: transform from halolet space to observed space
        itranspose: transpose of itransform operator

    """
    def __init__(self,parser,lensKernel):
        # transverse plane
        self.nframe =   parser.getint('sparse','nframe')
        self.ny     =   parser.getint('transPlane','ny')
        self.nx     =   parser.getint('transPlane','nx')
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
        self.ks2D   =   ksmap(self.ny,self.nx)

        # line of sight
        self.nzl    =   parser.getint('lens','nlp')
        self.nzs    =   parser.getint('sources','nz')
        if self.nzl <=  1:
            self.zlMin  =   0.
            self.zlscale=   1.
        else:
            self.zlMin  =   parser.getfloat('lens','zlMin')
            self.zlscale=   parser.getfloat('lens','zlscale')
        self.zlBin      =   zMeanBin(self.zlMin,self.zlscale,self.nzl)
        self.scale      =   parser.getfloat('transPlane','scale')*self.ratio
        self.sigma_pix  =   parser.getfloat('transPlane','smooth_scale')\
                            *self.ratio/self.scale

        # Shape of output shapelets
        self.shapeP =   (self.ny,self.nx)                   # basic plane
        self.shapeL =   (self.nzl,self.ny,self.nx)          # lens plane
        self.shapeA =   (self.nzl,self.nframe,self.ny,self.nx) # dictionary plane
        self.shapeS =   (self.nzs,self.ny,self.nx)          # observe plane
        if parser.has_option('lens','atomFname'):
            atFname =   parser.get('lens','atomFname')
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
            self.prepareFrames(parser)
        self.lensKernel=    lensKernel

    def prepareFrames(self,parser):
        if parser.has_option('cosmology','omega_m'):
            omega_m =   parser.getfloat('cosmology','omega_m')
        else:
            omega_m =   Default_OmegaM
        self.cosmo  =   Cosmo(H0=Default_h0*100.,Om0=omega_m)
        self.rs_base=   parser.getfloat('lens','rs_base')  # Mpc/h
        self.resolve_lim  =   parser.getfloat('lens','resolve_lim')
        # Initialize basis predictors
        # In configure Space
        self.aframes    =   np.zeros(self.shapeA,dtype=np.complex128)
        # In Fourier space
        self.fouaframes =   np.zeros(self.shapeA,dtype=np.complex128)
        # Intermediate basis in Fourier space
        self.fouaframesInter =   np.zeros(self.shapeA,dtype=np.complex128)
        self.rs_frame   =   -1.*np.ones((self.nzl,self.nframe)) # Radius in pixel

        for izl in range(self.nzl):
            # the r_s for each redshift plane in units of pixel
            rpix    =   self.cosmo.angular_diameter_distance(self.zlBin[izl]).value/180.*np.pi*self.scale
            rz      =   self.rs_base/rpix
            # nfw halo with mass normalized to 1e14
            znorm   =   1./rpix**2.
            # angular scale of pixel size in Mpc
            for ifr in reversed(range(self.nframe)):
                # For each lens redshift bins, we begin from the
                # frame with largest angular scale radius
                rs  =   (ifr+1)*rz
                if rs<self.resolve_lim:
                    # if one scale frame is less than resolution limit,
                    # skip this frame
                    break
                self.rs_frame[izl,ifr]= rs
                # nfw halo with mass normalized to 1e14
                iAtomF  =   haloCS02SigmaAtom(r_s=rs,ny=self.ny,nx=self.nx,c=4.,\
                            sigma_pix=self.sigma_pix)
                normTmp =   iAtomF[0,0]/znorm
                iAtomF  =   iAtomF/normTmp
                self.fouaframesInter[izl,ifr]=iAtomF        # Fourier Space
                iAtomF= self.ks2D.transform(iAtomF,inFou=True,outFou=True)
                # KS transform
                self.fouaframes[izl,ifr]=iAtomF             # Fourier Space
                self.aframes[izl,ifr]=np.fft.ifft2(iAtomF)  # Real Space
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
        Parameters:
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
