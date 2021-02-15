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
import numpy as np
from cosmology import Cosmo
import scipy.special as spfun
import astropy.io.fits as pyfits

#important constant
C_LIGHT=2.99792458e8        # m/s
GNEWTON=6.67428e-11         # m^3/kg/s^2
KG_PER_SUN=1.98892e30       # kg/M_solar
M_PER_PARSEC=3.08568025e16  # m/pc

def four_pi_G_over_c_squared():
    # = 1.5*H0^2/roh_0/c^2
    # We want it return 4piG/c^2 in unit of Mpc/M_solar
    # in unit of m/kg
    fourpiGoverc2 = 4.0*np.pi*GNEWTON/(C_LIGHT**2)
    # in unit of pc/M_solar
    fourpiGoverc2 *= KG_PER_SUN/M_PER_PARSEC
    # in unit of Mpc/M_solar
    fourpiGoverc2 /= 1.e6
    return fourpiGoverc2

class nfwHalo(Cosmo):
    def __init__(self,ra,dec,redshift,mass,conc=None,rs=None,omega_m=0.3):
        """
        @param mass         Mass defined using a spherical overdensity of 200 times the critical density
                            of the universe, in units of M_solar/h.
        @param conc         Concentration parameter, i.e., ratio of virial radius to NFW scale radius.
        @param redshift     Redshift of the halo.
        @param ra           ra of halo center  [arcsec].
        @param dec          dec of halo center [arcsec].
        @param omega_m      Omega_matter to pass to Cosmology constructor. [default: 0.3]
                            omega_lam is set to 1-omega_matter.
        """
        # Redshift and Geometry
        ## ra dec
        self.ra     =   ra
        self.dec    =   dec
        Cosmo.__init__(self,h=1,omega_m=omega_m)
        self.z      =   float(redshift)
        self.a      =   1./(1.+self.z)
        self.DaLens =   self.Da(0.,self.z) # angular distance in Mpc/h
        # E(z)^{-1}
        self.ezInv  =   self.Ez_inverse(self.z)
        # critical density
        # in unit of M_solar / Mpc^3
        rho_cZ      =   self.rho0()/self.ezInv**2
        self.M      =   float(mass)
        # First, we get the virial radius, which is defined for some spherical
        # overdensity as 3 M / [4 pi (r_vir)^3] = overdensity. Here we have
        # overdensity = 200 * rhocrit, to determine r_vir (angular distance).
        # The factor of 1.63e-5 comes from the following set of prefactors:
        # (3 / (4 pi * 200 * rhocrit))^(1/3), where rhocrit = 2.8e11 h^2
        # M_solar / Mpc^3.
        # (H0=100,DH=C_LIGHT/1e3/H0,rho_crit=1.5/four_pi_G_over_c_squared()/(DH)**2.)
        self.rvir        =   1.63e-5*(self.M*self.ezInv**2)**(1./3.) # in Mpc/h
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

        # \delta_c in equation (2)
        self.A      =   1./(np.log(1+self.c)-(self.c)/(1+self.c))
        self.delta_nfw =200./3*self.c**3*self.A
        # convert to angular radius in unit of arcsec
        scale       =   self.rs / self.DaLens
        arcsec2rad  =   np.pi/180./3600
        self.rs_arcsec =scale/arcsec2rad

        # Second, derive the charateristic matter density
        # within virial radius at redshift z
        self.rho_s=   rho_cZ*self.delta_nfw
        return

class nfw_lensWB00(nfwHalo):
    """
    Based on the integral functions of a spherical NFW profile:
    Wright & Brainerd(2000, ApJ, 534, 34)

    @param mass         Mass defined using a spherical overdensity of 200 times the critical density
                        of the universe, in units of M_solar/h.
    @param conc         Concentration parameter, i.e., ratio of virial radius to NFW scale radius.
    @param redshift     Redshift of the halo.
    @param ra           ra of halo center  [arcsec].
    @param dec          dec of halo center [arcsec].
    @param omega_m      Omega_matter to pass to Cosmology constructor. [default: 0.3]
                        omega_lam is set to 1-omega_matter.
    """
    def __init__(self,ra,dec,redshift,mass=None,conc=None,rs=None,omega_m=0.3):
        nfwHalo.__init__(self,ra,dec,redshift,mass=mass,conc=conc,rs=rs,omega_m=0.3)

    def __DdRs(self,ra_s,dec_s):
        """Calculate 'x' the radius r in units of the NFW scale
        radius, r_s.
        @param ra_s       ra of sources [arcsec].
        @param dec_s      dec of sources [arcsec].
        """
        x = ((ra_s - self.ra)**2 + (dec_s - self.dec)**2)**0.5/self.rs_arcsec
        return x

    def __sin2phi(self,ra_s,dec_s):
        """Calculate cos2phi and sin2phi with reference to the halo center
        @param ra_s       ra of sources [arcsec].
        @param dec_s      dec of sources [arcsec].
        """
        # pure tangential shear, no cross component
        dx = ra_s - self.ra
        dy = dec_s - self.dec
        drsq = dx*dx+dy*dy
        return np.divide(2*dx*dy, drsq, where=(drsq != 0.))

    def __cos2phi(self,ra_s,dec_s):
        """Calculate cos2phi and sin2phi with reference to the halo center
        @param ra_s       ra of sources [arcsec].
        @param dec_s      dec of sources [arcsec].
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
        @param ra_s       ra of sources [arcsec].
        @param dec_s      dec of sources [arcsec].
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
        mask = np.where(x <= 0.01)[0]
        out[mask] = 4*(0.25 + 0.125 * x[mask]**2 * (3.25 + 3.0*np.log(x[mask]/2)))*self.rs * self.rho_s
        return out

    def DeltaSigma(self,ra_s,dec_s):
        """Calculate excess surface density of halo.
        @param ra_s       ra of sources [arcsec].
        @param dec_s      dec of sources [arcsec].
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
        @param ra_s       ra of sources [arcsec].
        @param dec_s      dec of sources [arcsec].
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
        @param z_s        redshift of sources.
        """
        # convenience: call with single number
        if not isinstance(z_s, np.ndarray):
            return self.lensKernel(np.array([z_s], dtype='float'))[0]
        # lensing weights: the only thing that depends on z_s
        # First mask the data with z<z_l
        k_s =   np.zeros(len(z_s))
        mask=   z_s>self.z
        k_s[mask] =   self.Da(self.z,z_s[mask])*self.DaLens/self.Da(0.,z_s[mask])*four_pi_G_over_c_squared()
        return k_s

    def Sigma_M_bin(self,z_bin_min,z_bin_max):
        """Zero-order Surface mass density **background**
        within redshift bin [z_bin_min,z_bin_max].
        @param z_bin_min   minimum of redshift bin.
        @param z_bin_max   maximum of redshift bin.
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
        DaBin=self.Da(z_bin_min,z_bin_max)
        return rhoM_ave*DaBin

    def SigmaAtom(self,pix_scale,ngrid,xc=None,yc=None):
        """NFW Atom
        @param pix_scale    pixel sacle [arcsec]
        @param ngrid        number of pixels on x and y axis
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
        @param pix_scale    pixel sacle [arcsec]
        @param ngrid        number of pixels on x and y axis
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

    @param mass         Mass defined using a spherical overdensity of 200 times the critical density
                        of the universe, in units of M_solar/h.
    @param conc         Concentration parameter, i.e., ratio of virial radius to NFW scale radius.
    @param redshift     Redshift of the halo.
    @param ra           ra of halo center  [arcsec].
    @param dec          dec of halo center [arcsec].
    @param omega_m      Omega_matter to pass to Cosmology constructor. [default: 0.3]
                        omega_lam is set to 1-omega_matter.
    """
    def __init__(self,ra,dec,redshift,mass=None,conc=None,rs=None,omega_m=0.3):
        nfwHalo.__init__(self,ra,dec,redshift,mass=mass,conc=conc,rs=rs,omega_m=0.3)

    def __DdRs(self,ra_s,dec_s):
        """Calculate 'x' the radius r in units of the NFW scale
        radius, r_s.
        @param ra_s       ra of sources [arcsec].
        @param dec_s      dec of sources [arcsec].
        """
        x = ((ra_s - self.ra)**2 + (dec_s - self.dec)**2)**0.5/self.rs_arcsec
        return x

    def __sin2phi(self,ra_s,dec_s):
        """Calculate cos2phi and sin2phi with reference to the halo center
        @param ra_s       ra of sources [arcsec].
        @param dec_s      dec of sources [arcsec].
        """
        # pure tangential shear, no cross component
        dx = ra_s - self.ra
        dy = dec_s - self.dec
        drsq = dx*dx+dy*dy
        return np.divide(2*dx*dy, drsq, where=(drsq != 0.))

    def __cos2phi(self,ra_s,dec_s):
        """Calculate cos2phi and sin2phi with reference to the halo center
        @param ra_s       ra of sources [arcsec].
        @param dec_s      dec of sources [arcsec].
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
        Equation (11) in Wright & Brainerd (2000, ApJ, 534, 34).
        @param ra_s       ra of sources [arcsec].
        @param dec_s      dec of sources [arcsec].
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
        """Calculate excess surface density of halo.
        @param ra_s       ra of sources [arcsec].
        @param dec_s      dec of sources [arcsec].
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
        @param ra_s       ra of sources [arcsec].
        @param dec_s      dec of sources [arcsec].
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
        Lensing kernel of halo as function of source redshift.
        @param z_s        redshift of sources.
        """
        # convenience: call with single number
        if not isinstance(z_s, np.ndarray):
            return self.lensKernel(np.array([z_s], dtype='float'))[0]
        # lensing weights: the only thing that depends on z_s
        # First mask the data with z<z_l
        k_s =   np.zeros(len(z_s))
        mask=   z_s>self.z
        k_s[mask] =   self.Da(self.z,z_s[mask])*self.DaLens/self.Da(0.,z_s[mask])*four_pi_G_over_c_squared()
        return k_s

    def Sigma_M_bin(self,z_bin_min,z_bin_max):
        """Zero-order Surface mass density
        within redshift bin [z_bin_min,z_bin_max].
        @param z_bin_min   minimum of redshift bin.
        @param z_bin_max   maximum of redshift bin.
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
        DaBin=self.Da(z_bin_min,z_bin_max)
        return rhoM_ave*DaBin

    def SigmaAtom(self,pix_scale,ngrid,xc=None,yc=None):
        """NFW Sigma in a postage stamp
        @param pix_scale    pixel sacle [arcsec]
        @param ngrid        number of pixels on x and y axis
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
        @param pix_scale    pixel sacle [arcsec]
        @param ngrid        number of pixels on x and y axis
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
def haloCS02SigmaAtom(r_s,ny,nx=None,c=9.,smooth_scale=-1,fou=True,lnorm=2.):
    """
    Make haloTJ03 halo (l2 normalized by default) from Fourier space following
    CS02:
    https://arxiv.org/pdf/astro-ph/0206508.pdf -- Eq.(81) and Eq.(82)

    Parameters:
    -----------
    r_s     [float]
            scale radius (in unit of pixel).
    ny,nx   [int]
            number of pixel in y and x directions.
    c       [float]
            truncation ratio (concentration)
    fou     [bool]
            in Fourier space
    """
    if nx is None:
        nx=ny
    x,y=np.meshgrid(np.fft.fftfreq(nx),np.fft.fftfreq(ny))
    x*=(2*np.pi);y*=(2*np.pi)
    rT=np.sqrt(x**2+y**2)
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
    # Smoothing
    if smooth_scale>0.1:
        atom    =   atom*np.exp(-(rT*smooth_scale)**2./2.)
    # Real space?
    if fou:
        if lnorm>0.:
            norm=   (np.sum(atom**lnorm)/(nx*ny))**(1./lnorm)
        else:
            norm=   1.
    else:
        atom    =   np.real(np.fft.ifft2(atom))
        if lnorm>0.:
            norm=   (np.sum(atom**lnorm))**(1./lnorm)
        else:
            norm=   1.
    return atom/norm

def GausAtom(sigma,ny,nx=None,fou=True,lnorm=2.):
    """
    Normalized Gaussian in a postage stamp
    normalized by l2 norm
    Parameters:
    --------------
    ny:     int
    number of pixel y directions.
    nx:     int [default=None]
    number of pixel x directions.
    sigma:  float
    scale factortor of the Gaussian function
    """
    if nx is None:
        nx=ny
    if sigma>0.01:
        x,y =   np.meshgrid(np.fft.fftfreq(nx),np.fft.fftfreq(ny))
        if fou:
            x  *=   (2*np.pi);y*=(2*np.pi)
            rT  =   np.sqrt(x**2+y**2)
            fun =   np.exp(-(rT*sigma)**2./2.)
            if lnorm>0.:
                norm=   (np.sum(fun**lnorm)/(nx*ny))**(1./lnorm)
            else:
                norm=   1.
        else:
            x  *=   (nx);y*=(ny)
            rT  =   np.sqrt(x**2+y**2)
            fun =   1./np.sqrt(2.*np.pi)/sigma*np.exp(-(rT/sigma)**2./2.)
            if lnorm>0.:
                norm=   (np.sum(fun**lnorm))**(1./lnorm)
            else:
                norm=   1.
        return  fun/norm
    else:
        if fou:
            return np.ones((ny,nx))
        else:
            out =   np.zeros((ny,nx))
            out[0,0]=1
            return out

def ksInverse(gMap):
    gFouMap =   np.fft.fft2(gMap)
    e2phiF  =   e2phiFou(gFouMap.shape)
    kFouMap =   gFouMap/e2phiF*np.pi
    kMap    =   np.fft.ifft2(kFouMap)
    return kMap

def ksForward(kMap):
    kFouMap =   np.fft.fft2(kMap)
    e2phiF  =   e2phiFou(gFouMap.shape)
    gFouMap =   kFouMap*e2phiF/np.pi
    gMap    =   np.fft.ifft2(gFouMap)
    return gMap

def e2phiFou(shape):
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
