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
import astropy
import numpy as np
from .default import *
import scipy.special as spfun
from astropy.cosmology import FlatLambdaCDM as Cosmo

def mc2rs(mass,conc,redshift,omega_m=Default_OmegaM):
    """
    Get the scale radius of NFW halo from mass and redshift
    Parameters:
        mass:        Mass defined using a spherical overdensity of 200 times the critical density
                        of the universe, in units of M_solar/h.
        conc:        Concentration parameter, i.e., ratio of virial radius to NFW scale radius.
        redshift:    Redshift of the halo.
    """
    cosmo   =   Cosmo(H0=Default_h0*100.,Om0=omega_m)
    z       =   redshift
    a       =   1./(1.+z)
    # angular distance in Mpc/h
    DaLens  =   cosmo.angular_diameter_distance_z1z2(0.,z).value
    # E(z)^{-1}
    ezInv   =   cosmo.inv_efunc(z)
    # critical density (in unit of M_sun h^2 / Mpc^3)
    rho_cZ  =   cosmo.critical_density(self.z).to_value(unit=rho_unt)
    rvir    =   1.63e-5*(mass*ezInv**2)**(1./3.) # in Mpc/h
    rs      =   rvir/conc
    A       =   1./(np.log(1+conc)-(conc)/(1+conc))
    delta_nfw   =   200./3*conc**3*A
    # convert to angular radius in unit of arcsec
    scale       =   rs / DaLens
    arcmin2rad  =   np.pi/180./60.
    rs_arcmin   =   scale/arcmin2rad
    return rs_arcmin

class nfwHalo(Cosmo):
    def __init__(self,ra,dec,redshift,mass,conc=None,rs=None,omega_m=Default_OmegaM):
        """
        Parameters:
            mass:       Mass defined using a spherical overdensity of 200 times the critical density
                        of the universe, in units of M_solar/h.
            conc:       Concentration parameter, i.e., ratio of virial radius to NFW scale radius.
            redshift:   Redshift of the halo.
            ra:         ra of halo center  [arcsec].
            dec:        dec of halo center [arcsec].
            omega_m:    Omega_matter to pass to Cosmology constructor. [default: Default_OmegaM]
                        omega_lam is set to 1-omega_matter.
        """
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

class nfw_lensWB00(nfwHalo):
    """
    Based on the integral functions of a spherical NFW profile:
    Wright & Brainerd(2000, ApJ, 534, 34)
    Parameters:

        mass:       Mass defined using a spherical overdensity of 200 times the critical density
                    of the universe, in units of M_solar/h.
        conc:       Concentration parameter, i.e., ratio of virial radius to NFW scale radius.
        redshift:   Redshift of the halo.
        ra:         ra of halo center  [arcsec].
        dec:        dec of halo center [arcsec].
        omega_m:    Omega_matter to pass to Cosmology constructor. [default: 0.3]
                    (omega_lam is set to 1-omega_matter)
    """
    def __init__(self,ra,dec,redshift,mass=None,conc=None,rs=None,omega_m=0.3):
        nfwHalo.__init__(self,ra,dec,redshift,mass=mass,conc=conc,rs=rs,omega_m=0.3)

    def __DdRs(self,ra_s,dec_s):
        """Calculate 'x' the radius r in units of the NFW scale
        radius, r_s.
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        x = ((ra_s - self.ra)**2 + (dec_s - self.dec)**2)**0.5/self.rs_arcsec
        return x

    def __sin2phi(self,ra_s,dec_s):
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

    def __cos2phi(self,ra_s,dec_s):
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
        x=self.__DdRs(ra_s,dec_s)
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
        out[mask] = x[mask]**(-2)*(4.*np.log(x[mask]/2)+8.*np.arctan(a)/(x[mask]**2 - 1)**0.5)*self.rs * self.rho_s\
                    -self.__Sigma(x[mask])

        mask = np.where((x < 0.999) & (x> 0.01))[0]  # Equivalent but usually faster than `mask = (x < 0.999)`
        a = ((1.-x[mask])/(x[mask]+1.))**0.5
        out[mask] = x[mask]**(-2)*(4.*np.log(x[mask]/2)+8.*np.arctanh(a)/(1-x[mask]**2)**0.5)*self.rs * self.rho_s\
                    -self.__Sigma(x[mask])
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
        x=self.__DdRs(ra_s,dec_s)
        return self.__DeltaSigma(x)

    def DeltaSigmaComplex(self,ra_s,dec_s):
        """Calculate excess surface density of halo.
        return a complex array \Delta \Sigma_1+ i \Delta \Sigma_2
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
        x=self.__DdRs(ra_s,dec_s)
        DeltaSigma=self.__DeltaSigma(x)
        DeltaSigma1 = -DeltaSigma*self.__cos2phi(ra_s,dec_s)
        DeltaSigma2 = -DeltaSigma*self.__sin2phi(ra_s,dec_s)
        return DeltaSigma1+1j*DeltaSigma2

    def lensKernel(self,z_s):
        """Lensing kernel from surface density at lens redshfit to source redshift
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
        k_s[mask] =   self.angular_diameter_distance_z1z2(self.z,z_s[mask]).value\
                *self.DaLens/self.angular_diameter_distance_z1z2(0.,z_s[mask]).value\
                *four_pi_G_over_c_squared()
        return k_s

    def Sigma_M_bin(self,z_bin_min,z_bin_max):
        """Zero-order Surface mass density **background**
        within redshift bin [z_bin_min,z_bin_max].
        Parameters:
            z_bin_min:   minimum of redshift bin.
            z_bin_max:   maximum of redshift bin.
        """
        # convenience: call with single number
        assert isinstance(z_bin_min,np.ndarray)==isinstance(z_bin_max,np.ndarray),\
            'z_bin_min and z_bin_max do not have same type'

        if not isinstance(z_bin_min,np.ndarray):
            z_bin_minA=np.array([z_bin_min], dtype='float')
            z_bin_maxA=np.array([z_bin_max], dtype='float')
            return self.Sigma(z_bin_minA,z_bin_maxA)[0]
        assert len(z_bin_min)==len(z_bin_max),\
            'input ra and dec have different length '
        # Here we approximate the average rho_M for redshif bin
        # as the rho_M at the mean redshift
        rhoM_ave=self.rho_m((z_bin_min+z_bin_max)/2.)
        DaBin=self.angular_diameter_distance_z1z2(z_bin_min,z_bin_max)
        return rhoM_ave*DaBin

    def SigmaAtom(self,pix_scale,ngrid,xc=None,yc=None):
        """NFW Atom
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
        """NFW Atom
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

class nfw_lensTJ03(nfwHalo):
    """
    Based on the integral functions of a spherical NFW profile:
    Takada & Jain(2003, MNRAS, 340, 580) Eq.27 -- Surface Density,
    and Takada & Jain (2003, MNRAS, 344, 857) Eq.17 -- Excess Surface Density
    Parameters:
        mass:         Mass defined using a spherical overdensity of 200 times the critical density
                      of the universe, in units of M_solar/h.
        conc:         Concentration parameter, i.e., ratio of virial radius to NFW scale radius.
        redshift:     Redshift of the halo.
        ra:           ra of halo center  [arcsec].
        dec:          dec of halo center [arcsec].
        omega_m:      Omega_matter to pass to Cosmology constructor. [default: 0.3]
                      omega_lam is set to 1-omega_matter.
    """
    def __init__(self,ra,dec,redshift,mass=None,conc=None,rs=None,omega_m=0.3):
        nfwHalo.__init__(self,ra,dec,redshift,mass=mass,conc=conc,rs=rs,omega_m=0.3)

    def __DdRs(self,ra_s,dec_s):
        """Calculate 'x' the radius r in units of the NFW scale
        radius, r_s.
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        x = ((ra_s - self.ra)**2 + (dec_s - self.dec)**2)**0.5/self.rs_arcsec
        return x

    def __sin2phi(self,ra_s,dec_s):
        """Calculate cos2phi and sin2phi with reference to the halo center
        ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # pure tangential shear, no cross component
        dx = ra_s - self.ra
        dy = dec_s - self.dec
        drsq = dx*dx+dy*dy
        return np.divide(2*dx*dy, drsq, where=(drsq != 0.))

    def __cos2phi(self,ra_s,dec_s):
        """Calculate cos2phi and sin2phi with reference to the halo center
        ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # pure tangential shear, no cross component
        dx = ra_s - self.ra
        dy = dec_s - self.dec
        drsq = dx*dx+dy*dy
        return np.divide(dx*dx-dy*dy, drsq, where=(drsq != 0.))

    def __Sigma(self,x0):
        c   = np.float(self.c)
        out = np.zeros_like(x0, dtype=float)

        # 3 cases: x < 1-0.001, x > 1+0.001, and |x-1| < 0.001
        mask = np.where(x0 < 0.999)[0]
        x=x0[mask]
        out[mask] = -np.sqrt(c**2.-x**2.)/(1-x**2.)/(1+c)+1./(1-x**2.)**1.5*np.arccosh((x**2.+c)/x/(1.+c))

        mask = np.where((x0 > 1.001) & (x0<c))[0]
        x=x0[mask]
        out[mask] = -np.sqrt(c**2.-x**2.)/(1-x**2.)/(1+c)-1./(x**2.-1)**1.5*np.arccos((x**2.+c)/x/(1.+c))

        mask = np.where((x0 >= 0.999) & (x0 <= 1.001))[0]
        x=x0[mask]
        out[mask] = (-2.+c+c**2.)/(3.*np.sqrt(-1.+c)*(1+c)**(3./2))+((2.-c-4.*c**2.-2.*c**3.)*(x-1.))/(5.*np.sqrt(-1.+c)*(1+c)**(5/2.))

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
        x=self.__DdRs(ra_s,dec_s)
        return self.__Sigma(x)

    def __DeltaSigma(self,x0):
        c   = np.float(self.c)
        out = np.zeros_like(x0, dtype=float)

        # 4 cases:
        # x < 1-0.001,|x-1| <= 0.001
        # 1.001<x<=c, x>c

        mask = np.where(x0 < 0.0001)[0]
        out[mask]=1./2.

        mask = np.where((x0 < 0.999) & (x0>0.0001) )[0]
        x=x0[mask]
        out[mask] = (-2.*c+((2.-x**2.)*np.sqrt(c**2.-x**2.))/(1-x**2))/((1+c)*x**2.)+((2-3*x**2)*np.arccosh((c+x**2)/((1.+c)*x)))/(x**2*(1-x**2.)**1.5)\
            +(2*np.log(((1.+c)*x)/(c+np.sqrt(c**2-x**2))))/x**2

        mask = np.where((x0 > 1.001) & (x0< c))[0]
        x=x0[mask]
        out[mask] = (-2.*c+((2.-x**2.)*np.sqrt(c**2.-x**2.))/(1-x**2))/((1+c)*x**2.)-((2-3*x**2)*np.arccos((c+x**2)/((1.+c)*x)))/(x**2*(-1+x**2.)**1.5)\
            +(2*np.log(((1.+c)*x)/(c+np.sqrt(c**2-x**2))))/x**2

        mask = np.where((x0 >= 0.999) & (x0 <= 1.001))[0]
        x=x0[mask]
        out[mask] = (10*np.sqrt(-1.+c**2)+c*(-6-6*c+11*np.sqrt(-1.+c**2))+6*(1 + c)**2*np.log((1. + c)/(c +np.sqrt(-1.+c**2))))/(3.*(1+c)**2)-\
            (-1.+x)*((94 + c*(113 + 60*np.sqrt((-1.+c)/(1 + c))+4*c*(-22 + 30*np.sqrt((-1 + c)/(1 + c)) + c*(-26 + 15*np.sqrt((-1 + c)/(1 + c))))))/(15.*(1.+c)**2*np.sqrt(-1.+c**2))- 4*np.log(1.+c)+\
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
        x=self.__DdRs(ra_s,dec_s)
        return self.__DeltaSigma(x)

    def DeltaSigmaComplex(self,ra_s,dec_s):
        """Calculate excess surface density of halo.
        return a complex array \Delta \Sigma_1+ i \Delta \Sigma_2
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
        x=self.__DdRs(ra_s,dec_s)
        DeltaSigma=self.__DeltaSigma(x)
        DeltaSigma1 = -DeltaSigma*self.__cos2phi(ra_s,dec_s)
        DeltaSigma2 = -DeltaSigma*self.__sin2phi(ra_s,dec_s)
        return DeltaSigma1+1j*DeltaSigma2

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

    def Sigma_M_bin(self,z_bin_min,z_bin_max):
        """Zero-order Surface mass density within redshift bin
        [z_bin_min,z_bin_max].
        Parameters:
            z_bin_min:   minimum of redshift bin.
            z_bin_max:   maximum of redshift bin.
        """
        # convenience: call with single number
        assert isinstance(z_bin_min,np.ndarray)==isinstance(z_bin_max,np.ndarray),\
            'z_bin_min and z_bin_max do not have same type'

        if not isinstance(z_bin_min,np.ndarray):
            z_bin_minA=np.array([z_bin_min], dtype='float')
            z_bin_maxA=np.array([z_bin_max], dtype='float')
            return self.Sigma(z_bin_minA,z_bin_maxA)[0]
        assert len(z_bin_min)==len(z_bin_max),\
            'input ra and dec have different length '
        # Here we approximate the average rho_M for redshif bin
        # as the rho_M at the mean redshift
        rhoM_ave=self.rho_m((z_bin_min+z_bin_max)/2.)
        DaBin=self.angular_diameter_distance_z1z2(z_bin_min,z_bin_max)
        return rhoM_ave*DaBin

    def SigmaAtom(self,pix_scale,ngrid,xc=None,yc=None):
        """NFW Sigma in a postage stamp
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
        """NFW Delta Sigma in a postage stamp
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

"""
The following functions are used for halolet construction
"""
def haloCS02SigmaAtom(r_s,ny,nx=None,c=9.,sigma_pix=None,fou=True,lnorm=2.):
    """
    Make haloTJ03 halo (l2 normalized by default) from Fourier space following
    CS02:
    https://arxiv.org/pdf/astro-ph/0206508.pdf -- Eq.(81) and Eq.(82)

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

def GausAtom(sigma,ny,nx=None,fou=True,lnorm=2.):
    """
    Normalized Gaussian in a postage stamp
    Parameters:
        sigma:  std of Gaussian function; [float]
        ny:     number of pixel y directions; [int]
        nx:     number of pixel for x; [int, default=None]
        fou:    in Fourier/configuration space [bool, ture/false]
        lnorm:  normalized by lp norm [float]
    """
    if nx is None:
        nx=ny
    if sigma>0.01:
        x,y =   np.meshgrid(np.fft.fftfreq(nx),np.fft.fftfreq(ny))
        norm=   1. # an initialization
        if fou:
            # Fourier space
            x  *=   (2*np.pi);y*=(2*np.pi)
            rT  =   np.sqrt(x**2+y**2)
            fun =   np.exp(-(rT*sigma)**2./2.)
            if lnorm>0.:
                norm=   (np.sum(fun**lnorm)/(nx*ny))**(1./lnorm)
        else:
            # Configuration space
            x  *=   (nx);y*=(ny)
            rT  =   np.sqrt(x**2+y**2)
            fun =   1./np.sqrt(2.*np.pi)/sigma*np.exp(-(rT/sigma)**2./2.)
            if lnorm>0.:
                norm=   (np.sum(fun**lnorm))**(1./lnorm)
        return  fun/norm
    else:
        if fou:
            return np.ones((ny,nx))
        else:
            out =   np.zeros((ny,nx))
            out[0,0]=1
            return out

def TophatAtom(width,ny,nx=None,fou=True,lnorm=-1):
    """
    Normalized top-hat atom in a postage stamp
    Parameters:
        width:  width of top-hat function(in unit of pixel)
        ny:     number of pixel y directions; [int]
        nx:     number of pixel for x; [int, default=None]
        fou:    in Fourier/configuration space [bool, ture/false]
        lnorm:  normalized by lp norm [float]
    """
    if nx is None:
        nx=ny
    assert width>0.9 and width<nx-4 and width<ny-4
    width=int(width+0.5)
    norm=1.
    if fou:
        x,y =   np.meshgrid(np.fft.fftfreq(nx),np.fft.fftfreq(ny))
        x  *=   (2*np.pi);y*=(2*np.pi)
        funx = np.divide(np.sin(width*x/2),(width*x/2),out=np.ones_like(x), where=x!=0)
        funy = np.divide(np.sin(width*y/2),(width*y/2),out=np.ones_like(y), where=y!=0)
        fun  = funx*funy
        if lnorm>0.:
            norm=   (np.sum(fun**lnorm)/(nx*ny))**(1./lnorm)
    else:
        sx=(nx-width)//2;sy=(ny-width)//2
        fun=np.zeros((ny,nx))
        fun[sy+1:sy+width,sx+1:sx+width]=1./width**2.
    return  fun/norm
