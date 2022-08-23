# Copyright 20220706 Xiangchong Li & Shouzhuo Yang.
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
from scipy.misc import derivative
from scipy.integrate import quad
import pyfftw
from astropy.io import fits
from astropy.table import Table


# import random


def zMeanBin(zMin, dz, nz):
    return np.arange(zMin, zMin + dz * nz, dz) + dz / 2.


def haloCS02SigmaAtom(r_s, ny, nx=None, c=9., sigma_pix=None, fou=True, lnorm=2., truncate=True):
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
        nx = ny
    x, y = np.meshgrid(np.fft.fftfreq(nx), np.fft.fftfreq(ny))
    x *= (2 * np.pi);
    y *= (2 * np.pi)
    rT = np.sqrt(x ** 2 + y ** 2)
    if r_s <= 0.1:
        # point mass in Fourier space
        atom = np.ones((ny, nx))
    else:
        # NFW halo in Fourier space
        A = 1. / (np.log(1 + c) - c / (1. + c))
        r = rT * r_s
        mask = r > 0.001
        atom = np.zeros_like(r, dtype=float)
        r1 = r[mask]
        si1, ci1 = spfun.sici((1 + c) * r1)
        si2, ci2 = spfun.sici(r1)
        r0 = r[~mask]
        if truncate:
            # original version of NFW in fourier space.
            atom[mask] = A * (np.sin(r1) * (si1 - si2) - np.sin(c * r1) / (1 + c) / r1 + np.cos(r1) * (ci1 - ci2))
            atom[~mask] = 1. + A * (
                    c + c ** 3 / (6 * (1 + c)) + 1 / 4. * (-2. * c - c ** 2. - 2 * np.log(1 + c))) * r0 ** 2.
        else:
            atom[mask] = 1 / 2 * (-np.cos(r1) * 2 * ci2 + np.sin(r1) * (np.pi - 2 * si2))
            # Idk the meaning of A so I am leaving it out for now.
            # because we select r>0.001, points closest to the halo's center have r on the magnitude of ~0.0005
            mock_small_distance = np.ones_like(r0, dtype=float) * 0.0005
            atom[~mask] = -np.euler_gamma - np.log(mock_small_distance)
    if sigma_pix is not None:
        if sigma_pix > 0.1:
            # Gaussian smoothing
            atom = atom * np.exp(-(rT * sigma_pix) ** 2. / 2.)
        else:
            # top-hat smoothing
            atom = atom * TophatAtom(width=1., ny=ny, nx=nx, fou=True)

    if fou:
        # Fourier space
        if lnorm > 0.:
            norm = (np.sum(atom ** lnorm) / (nx * ny)) ** (1. / lnorm)
        else:
            norm = 1.
    else:
        # configuration space
        atom = np.real(np.fft.ifft2(atom))
        if lnorm > 0.:
            norm = (np.sum(atom ** lnorm)) ** (1. / lnorm)
        else:
            norm = 1.
    return atom / norm


def mc2rs(mass, conc, redshift, omega_m=Default_OmegaM):
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
    cosmo = Cosmo(H0=Default_h0 * 100., Om0=omega_m)
    z = redshift
    # a       =   1./(1.+z)
    # angular distance in Mpc/h
    DaLens = cosmo.angular_diameter_distance_z1z2(0., z).value
    # E(z)^{-1}
    ezInv = cosmo.inv_efunc(z)
    # critical density (in unit of M_sun h^2 / Mpc^3)
    # rho_cZ  =   cosmo.critical_density(self.z).to_value(unit=rho_unt)
    rvir = 1.63e-5 * (mass * ezInv ** 2) ** (1. / 3.)  # in Mpc/h
    rs = rvir / conc
    # A       =   1./(np.log(1+conc)-(conc)/(1+conc))
    # delta_nfw   =   200./3*conc**3*A
    # convert to angular radius in unit of arcsec
    scale = rs / DaLens
    arcmin2rad = np.pi / 180. / 60.
    rs_arcmin = scale / arcmin2rad
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

    def __init__(self, ny, nx):
        self.shape = (ny, nx)
        self.e2phiF = self.e2phiFou()
        self.a = pyfftw.empty_aligned(self.shape, dtype='complex128')
        self.b = pyfftw.empty_aligned(self.shape, dtype='complex128')
        self.fft_object_forward = pyfftw.FFTW(self.a, self.b, axes=(0, 1))
        self.fft_object_inverse = pyfftw.FFTW(self.a, self.b, axes=(0, 1), direction='FFTW_BACKWARD')

    def e2phiFou(self):
        ny, nx = self.shape
        e2phiF = np.zeros(self.shape, dtype=np.complex128)
        for j in range(ny):
            jy = (j + ny // 2.) % ny - ny // 2.
            jy = np.float64(jy / ny)
            for i in range(nx):
                ix = (i + nx // 2.) % nx - nx // 2.
                ix = np.float64(ix / nx)
                if i == 0 and j == 0:
                    e2phiF[j, i] = 0.
                else:
                    r2 = ix ** 2. + jy ** 2.
                    e2phiF[j, i] = (ix ** 2. - jy ** 2.) / r2 + (2j * ix * jy / r2)
        return e2phiF * np.pi

    def itransform(self, gMap, inFou=True, outFou=True):
        """
        K-S Transform from gamma map to kappa map

        Parameters:
        kMap:   input gamma map
        inFou:  input in Fourier space? [default:True=yes]
        outFou: output in Fourier space? [default:True=yes]
        """
        assert gMap.shape[-2:] == self.shape
        if not inFou:
            # gMap =   np.fft.fft2(gMap)
            gMap = self.fft_object_forward(gMap)
        kOMap = gMap * np.conjugate(self.e2phiF * np.pi)
        if not outFou:
            # kOMap    =   np.fft.ifft2(kOMap)
            kOMap = self.fft_object_inverse(kOMap)
        return kOMap

    def transform_fftw(self, kMap, inFou=True, outFou=True):
        """
        K-S Transform from kappa map to gamma map

        Parameters:
        gMap:   input kappa map
        inFou:  input in Fourier space? [default:True=yes]
        outFou: output in Fourier space? [default:True=yes]
        """
        assert kMap.shape[-2:] == self.shape
        if not inFou:
            kMap = self.fft_object_forward(kMap)
        gOMap = kMap * self.e2phiF / np.pi
        if not outFou:
            gOMap = self.fft_object_inverse(gOMap)
        return gOMap

    def transform(self, kMap, inFou=True, outFou=True):
        """
        K-S Transform from kappa map to gamma map

        Parameters:
        gMap:   input kappa map
        inFou:  input in Fourier space? [default:True=yes]
        outFou: output in Fourier space? [default:True=yes]
        """
        assert kMap.shape[-2:] == self.shape
        if not inFou:
            # kMap =   np.fft.fft2(kMap)
            kMap = self.fft_object_forward(kMap)
        gOMap = kMap * self.e2phiF / np.pi
        if not outFou:
            # gOMap    =   np.fft.ifft2(gOMap)
            gOMap = self.fft_object_inverse(gOMap)
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

    def __init__(self, ra, dec, redshift, mass, conc=None, rs=None, omega_m=Default_OmegaM):
        # Redshift and Geometry
        ## ra dec
        self.ra = ra
        self.dec = dec
        Cosmo.__init__(self, H0=Default_h0 * 100., Om0=omega_m)
        self.z = float(redshift)
        self.a = 1. / (1. + self.z)
        # angular distance in Mpc/h
        self.DaLens = self.angular_diameter_distance_z1z2(0., self.z).value
        # critical density
        # in unit of M_solar / Mpc^3
        rho_cZ = self.critical_density(self.z).to_value(unit=rho_unt)
        self.rho_cZ = rho_cZ

        self.M = float(mass)
        ezInv = self.inv_efunc(redshift)
        self.ezInv = ezInv
        self.Omega_z = self.Om(self.z)
        self.omega_vir = 1 / (self.Om0 * (1 + self.z) ** 3 / (self.Om0 * (1 + self.z) ** 3 + 1 - self.Om0)) - 1
        self.Delta_vir = 18 * np.pi ** 2 * (1 + 0.4093 * self.omega_vir ** (0.9052))
        self.rvir = (3 * self.M / (4 * np.pi * self.Delta_vir * self.Omega_z * self.rho_cZ)) ** (1 / 3)
        # First, we get the virial radius, which is defined for some spherical
        # overdensity as 3 M / [4 pi (r_vir)^3] = overdensity. Here we have
        # overdensity = 200 * rhocrit, to determine r_vir (angular distance).
        # The factor of 1.63e-5 [h^(-2/3.)] comes from the following set of prefactors:
        # (3 / (4 pi * 200 * rhocrit))^(1/3), where rhocrit = 2.8e11 h^2
        # M_solar / Mpc^3.
        if conc is not None:
            self.c = float(conc)
            # scale radius
            self.rs = self.rvir / self.c
            if rs is not None:
                assert abs(self.rs - rs) < 0.01, 'input rs is different from derived'
        elif rs is not None:
            self.rs = float(rs)
            self.c = self.rvir / self.rs
        else:
            raise ValueError("need to give conc or rs, at least one")
        self.m_c = np.log(1 + self.c) - self.c / (1 + self.c)  # OLS eqn 9, alpha = 1
        self.A = 1 / self.m_c
        self.delta_nfw = self.Delta_vir * self.Omega_z / 3 * self.c ** 3 / self.m_c  # with spherical model
        # Delta_vir = Delta_e
        # convert to angular radius in unit of arcsec
        scale = self.rs / self.DaLens
        arcsec2rad = np.pi / 180. / 3600.
        self.rs_arcsec = scale / arcsec2rad

        # Second, derive the charateristic matter density
        # within virial radius at redshift z
        self.rho_s = rho_cZ * self.delta_nfw

        return

    def DdRs(self, ra_s, dec_s):
        """Calculate 'x' the radius r in units of the NFW scale
        radius, r_s.
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        x = ((ra_s - self.ra) ** 2 + (dec_s - self.dec) ** 2) ** 0.5 / self.rs_arcsec
        return x

    def sin2phi(self, ra_s, dec_s):
        """Calculate cos2phi and sin2phi with reference to the halo center
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # pure tangential shear, no cross component
        dx = ra_s - self.ra
        dy = dec_s - self.dec
        drsq = dx * dx + dy * dy
        return np.divide(2 * dx * dy, drsq, where=(drsq != 0.))

    def cos2phi(self, ra_s, dec_s):
        """Calculate cos2phi and sin2phi with reference to the halo center
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # pure tangential shear, no cross component
        dx = ra_s - self.ra
        dy = dec_s - self.dec
        drsq = dx * dx + dy * dy
        return np.divide(dx * dx - dy * dy, drsq, where=(drsq != 0.))

    def lensKernel(self, z_s):
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
        k_s = np.zeros(len(z_s))
        mask = z_s > self.z
        k_s[mask] = self.angular_diameter_distance_z1z2(self.z, z_s[mask]) * self.DaLens \
                    / self.angular_diameter_distance_z1z2(0., z_s[mask]) * four_pi_G_over_c_squared()
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

    def __init__(self, ra, dec, redshift, mass=None, conc=None, rs=None, omega_m=Default_OmegaM, long_truncation=False):
        nfwHalo.__init__(self, ra, dec, redshift, mass=mass, conc=conc, rs=rs, omega_m=omega_m)
        self.long_truncation = long_truncation

    def __Sigma(self, x):
        out = np.zeros_like(x, dtype=float)

        # 3 cases: x < 1, x > 1, and |x-1| < 0.001
        mask = np.where(x < 0.999)[0]
        a = ((1 - x[mask]) / (x[mask] + 1)) ** 0.5
        out[mask] = 2 / (x[mask] ** 2 - 1) * (1 - 2 * np.arctanh(a) / (1 - x[mask] ** 2) ** 0.5)

        mask = np.where(x > 1.001)[0]
        a = ((x[mask] - 1) / (x[mask] + 1)) ** 0.5
        out[mask] = 2 / (x[mask] ** 2 - 1) * (1 - 2 * np.arctan(a) / (x[mask] ** 2 - 1) ** 0.5)

        # the approximation below has a maximum fractional error of 7.4e-7
        mask = np.where((x >= 0.999) & (x <= 1.001))[0]
        out[mask] = (22. / 15. - 0.8 * x[mask])
        if self.long_truncation:
            # mask = np.where(x > 2 * self.c)[0]
            mask = np.where(x > np.inf)[0]
        else:
            mask = np.where(x > self.c)[0]
        out[mask] = 0
        return out * self.rs * self.rho_s

    def Sigma(self, ra_s, dec_s):
        """Calculate Surface Density (Sigma) of halo.
        Equation (11) in Wright & Brainerd (2000, ApJ, 534, 34).
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # convenience: call with single number
        assert isinstance(ra_s, np.ndarray) == isinstance(dec_s, np.ndarray), \
            'ra_s and dec_s do not have same type'
        if not isinstance(ra_s, np.ndarray):
            ra_sA = np.array([ra_s], dtype='float')
            dec_sA = np.array([dec_s], dtype='float')
            return self.Sigma(ra_sA, dec_sA)[0]
        assert len(ra_s) == len(dec_s), \
            'input ra and dec have different length '
        x = self.DdRs(ra_s, dec_s)
        return self.__Sigma(x)

    def __DeltaSigma(self, x):
        out = np.zeros_like(x, dtype=float)
        """
        # 4 cases:
        # x > 1,0.01< x < 1,|x-1| < 0.001
        # x<0.01
        """
        mask = np.where(x > 1.001)[0]
        a = ((x[mask] - 1.) / (x[mask] + 1.)) ** 0.5
        out[mask] = x[mask] ** (-2) * (4. * np.log(x[mask] / 2) + 8. * np.arctan(a) \
                                       / (x[mask] ** 2 - 1) ** 0.5) * self.rs * self.rho_s - self.__Sigma(x[mask])
        # Equivalent but usually faster than mask = (x < 0.999)
        mask = np.where((x < 0.999) & (x > 0.01))[0]
        a = ((1. - x[mask]) / (x[mask] + 1.)) ** 0.5
        out[mask] = x[mask] ** (-2) * (4. * np.log(x[mask] / 2) + 8. * np.arctanh(a) \
                                       / (1 - x[mask] ** 2) ** 0.5) * self.rs * self.rho_s - self.__Sigma(x[mask])
        """
        # the approximation below has a maximum fractional error of 2.3e-7
        """
        mask = np.where((x >= 0.999) & (x <= 1.001))[0]
        out[mask] = (4. * np.log(x[mask] / 2) + 40. / 6. - 8. * x[mask] / 3.) * self.rs * self.rho_s \
                    - self.__Sigma(x[mask])
        """
        # the approximation below has a maximum fractional error of 1.1e-7
        """
        mask = np.where(x <= 0.01)[0]
        out[mask] = 4. * (0.25 + 0.125 * x[mask] ** 2 * \
                          (3.25 + 3.0 * np.log(x[mask] / 2))) * self.rs * self.rho_s
        return out

    def DeltaSigma(self, ra_s, dec_s):
        """Calculate excess surface density of halo.
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # convenience: call with single number
        assert isinstance(ra_s, np.ndarray) == isinstance(dec_s, np.ndarray), \
            'ra_s and dec_s do not have same type'
        if not isinstance(ra_s, np.ndarray):
            ra_sA = np.array([ra_s], dtype='float')
            dec_sA = np.array([dec_s], dtype='float')
            return self.DeltaSigma(ra_sA, dec_sA)[0]
        assert len(ra_s) == len(dec_s), \
            'input ra and dec have different length '
        x = self.DdRs(ra_s, dec_s)
        return self.__DeltaSigma(x)

    def DeltaSigmaComplex(self, ra_s, dec_s):
        """Calculate excess surface density of halo.
        return a complex array Delta Sigma_1+ i Delta Sigma_2
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # convenience: call with single number
        assert isinstance(ra_s, np.ndarray) == isinstance(dec_s, np.ndarray), \
            'ra_s and dec_s do not have same type'
        if not isinstance(ra_s, np.ndarray):
            ra_sA = np.array([ra_s], dtype='float')
            dec_sA = np.array([dec_s], dtype='float')
            return self.DeltaSigmaComplex(ra_sA, dec_sA)[0]
        assert len(ra_s) == len(dec_s), \
            'input ra and dec have different length '
        x = self.DdRs(ra_s, dec_s)
        DeltaSigma = self.__DeltaSigma(x)
        DeltaSigma1 = -DeltaSigma * self.cos2phi(ra_s, dec_s)
        DeltaSigma2 = -DeltaSigma * self.sin2phi(ra_s, dec_s)
        return DeltaSigma1 + 1j * DeltaSigma2

    def SigmaAtom(self, pix_scale, ngrid, xc=None, yc=None):
        """NFW Atom on Grid
        Parameters:
            pix_scale:    pixel sacle [arcsec]
            ngrid:        number of pixels on x and y axis
        """
        if xc is None:
            xc = self.ra
        if yc is None:
            yc = self.dec

        X = (np.arange(ngrid) - ngrid / 2.) * pix_scale + xc
        Y = (np.arange(ngrid) - ngrid / 2.) * pix_scale + yc
        x, y = np.meshgrid(X, Y)
        atomReal = self.Sigma(x.ravel(), y.ravel()).reshape((ngrid, ngrid))
        return atomReal

    def DeltaSigmaAtom(self, pix_scale, ngrid, xc=None, yc=None):
        """NFW Atom on Grid
        Parameters:
            pix_scale:    pixel sacle [arcsec]
            ngrid:        number of pixels on x and y axis
        """
        if xc is None:
            xc = self.ra
        if yc is None:
            yc = self.dec

        X = (np.arange(ngrid) - ngrid / 2.) * pix_scale + xc
        Y = (np.arange(ngrid) - ngrid / 2.) * pix_scale + yc
        x, y = np.meshgrid(X, Y)
        atomReal = self.DeltaSigma(x.ravel(), y.ravel()).reshape((ngrid, ngrid))
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

    def __init__(self, ra, dec, redshift, mass=None, conc=None, rs=None, omega_m=Default_OmegaM):
        nfwHalo.__init__(self, ra, dec, redshift, mass=mass, conc=conc, rs=rs, omega_m=omega_m)

    def __Sigma(self, x0):
        c = float(self.c)
        out = np.zeros_like(x0, dtype=float)

        # 3 cases: x < 1-0.001, x > 1+0.001, and |x-1| < 0.001
        mask = np.where(x0 < 0.999)[0]
        x = x0[mask]
        out[mask] = -np.sqrt(c ** 2. - x ** 2.) / (1 - x ** 2.) / (1 + c) + \
                    1. / (1 - x ** 2.) ** 1.5 * np.arccosh((x ** 2. + c) / x / (1. + c))

        mask = np.where((x0 > 1.001) & (x0 < c))[0]
        x = x0[mask]
        out[mask] = -np.sqrt(c ** 2. - x ** 2.) / (1 - x ** 2.) / (1 + c) - \
                    1. / (x ** 2. - 1) ** 1.5 * np.arccos((x ** 2. + c) / x / (1. + c))

        mask = np.where((x0 >= 0.999) & (x0 <= 1.001))[0]
        x = x0[mask]
        out[mask] = (-2. + c + c ** 2.) / (3. * np.sqrt(-1. + c) * (1 + c) ** (3. / 2)) \
                    + ((2. - c - 4. * c ** 2. - 2. * c ** 3.) * (x - 1.)) / (
                            5. * np.sqrt(-1. + c) * (1 + c) ** (5 / 2.))

        mask = np.where(x0 >= c)[0]
        out[mask] = 0.
        return out * self.rs * self.rho_s * 2.

    def Sigma(self, ra_s, dec_s):
        """Calculate Surface Density (Sigma) of halo.
        Takada & Jain(2003, MNRAS, 340, 580) Eq.27
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """

        # convenience: call with single number
        assert isinstance(ra_s, np.ndarray) == isinstance(dec_s, np.ndarray), \
            'ra_s and dec_s do not have same type'
        if not isinstance(ra_s, np.ndarray):
            ra_sA = np.array([ra_s], dtype='float')
            dec_sA = np.array([dec_s], dtype='float')
            return self.Sigma(ra_sA, dec_sA)[0]
        assert len(ra_s) == len(dec_s), \
            'input ra and dec have different length '
        x = self.DdRs(ra_s, dec_s)
        return self.__Sigma(x)

    def __DeltaSigma(self, x0):
        c = float(self.c)
        out = np.zeros_like(x0, dtype=float)

        # 4 cases:
        # x < 1-0.001,|x-1| <= 0.001
        # 1.001<x<=c, x>c

        mask = np.where(x0 < 0.0001)[0]
        out[mask] = 1. / 2.

        mask = np.where((x0 < 0.999) & (x0 > 0.0001))[0]
        x = x0[mask]
        out[mask] = (-2. * c + ((2. - x ** 2.) * np.sqrt(c ** 2. - x ** 2.)) / (1 - x ** 2)) / ((1 + c) * x ** 2.) \
                    + ((2 - 3 * x ** 2) * np.arccosh((c + x ** 2) / ((1. + c) * x))) / (x ** 2 * (1 - x ** 2.) ** 1.5) \
                    + (2 * np.log(((1. + c) * x) / (c + np.sqrt(c ** 2 - x ** 2)))) / x ** 2

        mask = np.where((x0 > 1.001) & (x0 < c))[0]
        x = x0[mask]
        out[mask] = (-2. * c + ((2. - x ** 2.) * np.sqrt(c ** 2. - x ** 2.)) / (1 - x ** 2)) / ((1 + c) * x ** 2.) \
                    - ((2 - 3 * x ** 2) * np.arccos((c + x ** 2) / ((1. + c) * x))) / (x ** 2 * (-1 + x ** 2.) ** 1.5) \
                    + (2 * np.log(((1. + c) * x) / (c + np.sqrt(c ** 2 - x ** 2)))) / x ** 2

        mask = np.where((x0 >= 0.999) & (x0 <= 1.001))[0]
        x = x0[mask]
        out[mask] = (10 * np.sqrt(-1. + c ** 2) + c * (-6 - 6 * c + 11 * np.sqrt(-1. + c ** 2)) \
                     + 6 * (1 + c) ** 2 * np.log((1. + c) / (c + np.sqrt(-1. + c ** 2)))) / (3. * (1 + c) ** 2) - \
                    (-1. + x) * ((94 + c * (
                113 + 60 * np.sqrt((-1. + c) / (1 + c)) + 4 * c * (-22 + 30 * np.sqrt((-1 + c) / (1 + c)) \
                                                                   + c * (-26 + 15 * np.sqrt(
                    (-1 + c) / (1 + c)))))) / (15. * (1. + c) ** 2 * np.sqrt(-1. + c ** 2)) - 4 * np.log(1. + c) + \
                                 4 * np.log(c + np.sqrt(-1. + c ** 2)))

        mask = np.where(x0 >= c)[0]
        x = x0[mask]
        out[mask] = 2. / self.A / x ** 2.
        return out * self.rs * self.rho_s * 2.

    def DeltaSigma(self, ra_s, dec_s):
        """Calculate excess surface density of halo according to
        Takada & Jain (2003, MNRAS, 344, 857) Eq.17 -- Excess Surface Density
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # convenience: call with single number
        assert isinstance(ra_s, np.ndarray) == isinstance(dec_s, np.ndarray), \
            'ra_s and dec_s do not have same type'
        if not isinstance(ra_s, np.ndarray):
            ra_sA = np.array([ra_s], dtype='float')
            dec_sA = np.array([dec_s], dtype='float')
            return self.DeltaSigma(ra_sA, dec_sA)[0]
        assert len(ra_s) == len(dec_s), \
            'input ra and dec have different length '
        x = self.DdRs(ra_s, dec_s)
        return self.__DeltaSigma(x)

    def DeltaSigmaComplex(self, ra_s, dec_s):
        """Calculate excess surface density of halo.
        return a complex array Delta Sigma_1+ i Delta Sigma_2
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # convenience: call with single number
        assert isinstance(ra_s, np.ndarray) == isinstance(dec_s, np.ndarray), \
            'ra_s and dec_s do not have same type'
        if not isinstance(ra_s, np.ndarray):
            ra_sA = np.array([ra_s], dtype='float')
            dec_sA = np.array([dec_s], dtype='float')
            return self.DeltaSigmaComplex(ra_sA, dec_sA)[0]
        assert len(ra_s) == len(dec_s), \
            'input ra and dec have different length '
        x = self.DdRs(ra_s, dec_s)
        DeltaSigma = self.__DeltaSigma(x)
        DeltaSigma1 = -DeltaSigma * self.cos2phi(ra_s, dec_s)
        DeltaSigma2 = -DeltaSigma * self.sin2phi(ra_s, dec_s)
        return DeltaSigma1 + 1j * DeltaSigma2

    def SigmaAtom(self, pix_scale, ngrid, xc=None, yc=None):
        """NFW Sigma on Grid
        Parameters:
            pix_scale:    pixel sacle [arcsec]
            ngrid:        number of pixels on x and y axis
        """
        if xc is None:
            xc = self.ra
        if yc is None:
            yc = self.dec

        X = (np.arange(ngrid) - ngrid / 2.) * pix_scale + xc
        Y = (np.arange(ngrid) - ngrid / 2.) * pix_scale + yc
        x, y = np.meshgrid(X, Y)
        atomReal = self.Sigma(x.ravel(), y.ravel()).reshape((ngrid, ngrid))
        return atomReal

    def DeltaSigmaAtom(self, pix_scale, ngrid, xc=None, yc=None):
        """NFW Delta Sigma on Grid
        Parameters:
            pix_scale:    pixel sacle [arcsec]
            ngrid:        number of pixels on x and y axis
        """
        if xc is None:
            xc = self.ra
        if yc is None:
            yc = self.dec

        X = (np.arange(ngrid) - ngrid / 2.) * pix_scale + xc
        Y = (np.arange(ngrid) - ngrid / 2.) * pix_scale + yc
        x, y = np.meshgrid(X, Y)
        atomReal = self.DeltaSigma(x.ravel(), y.ravel()).reshape((ngrid, ngrid))
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

    def __init__(self, parser):
        Cartesian.__init__(self, parser)
        self.ks2D = ksmap(self.ny, self.nx)
        return

    def add_halo(self, halo):
        lk = halo.lensKernel(self.zcgrid)
        rpix = halo.rs_arcsec / self.scale / 3600.

        sigma = haloCS02SigmaAtom(rpix, ny=self.ny, nx=self.nx, \
                                  sigma_pix=self.sigma_pix, c=halo.c, fou=True)
        snorm = sigma[0, 0]
        dr = halo.DaLens * self.scale / 180 * np.pi
        snorm = halo.M / dr ** 2. / snorm
        sigma = sigma * snorm
        dsigma = np.fft.fftshift(self.ks2D.transform(sigma, \
                                                     inFou=True, outFou=False))
        sigma = np.fft.fftshift(np.fft.ifft2(sigma)).real
        shear = dsigma[None, :, :] * lk[:, None, None]
        kappa = sigma[None, :, :] * lk[:, None, None]
        return kappa, shear, sigma, lk, snorm, dr


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

    def __init__(self, parser, lensKernel):
        # transverse plane
        self.nframe = parser.getint('sparse', 'nframe')
        self.ny = parser.getint('transPlane', 'ny')
        self.nx = parser.getint('transPlane', 'nx')

        # The unit of angle in the configuration
        unit = parser.get('transPlane', 'unit')
        # Rescaling to degree
        if unit == 'degree':
            self.ratio = 1.
        elif unit == 'arcmin':
            self.ratio = 1. / 60.
        elif unit == 'arcsec':
            self.ratio = 1. / 60. / 60.
        self.scale = parser.getfloat('transPlane', 'scale') * self.ratio
        self.ks2D = ksmap(self.ny, self.nx)

        # line of sight
        self.nzl = parser.getint('lens', 'nlp')
        self.nzs = parser.getint('sources', 'nz')
        if self.nzl <= 1:
            self.zlMin = 0.
            self.zlscale = 1.
        else:
            self.zlMin = parser.getfloat('lens', 'zlMin')
            self.zlscale = parser.getfloat('lens', 'zlscale')
        self.zlBin = zMeanBin(self.zlMin, self.zlscale, self.nzl)
        self.scale = parser.getfloat('transPlane', 'scale') * self.ratio
        self.sigma_pix = parser.getfloat('transPlane', 'smooth_scale') \
                         * self.ratio / self.scale

        # Shape of output shapelets
        self.shapeP = (self.ny, self.nx)  # basic plane
        self.shapeL = (self.nzl, self.ny, self.nx)  # lens plane
        self.shapeA = (self.nzl, self.nframe, self.ny, self.nx)  # dictionary plane
        self.shapeS = (self.nzs, self.ny, self.nx)  # observe plane

        ####pyfftw stuff####
        array1 = pyfftw.empty_aligned(self.shapeP, dtype='complex128', n=16)  # need to intialize pyfftw obj
        array2 = pyfftw.empty_aligned(self.shapeP, dtype='complex128', n=16)
        array3 = pyfftw.empty_aligned(self.shapeL, dtype='complex128', n=16)  # need to intialize pyfftw obj
        array4 = pyfftw.empty_aligned(self.shapeL, dtype='complex128', n=16)
        array5 = pyfftw.empty_aligned(self.shapeA, dtype='complex128', n=16)  # need to intialize pyfftw obj
        array6 = pyfftw.empty_aligned(self.shapeA, dtype='complex128', n=16)

        self.fftw2 = pyfftw.FFTW(array1, array2, axes=(0, 1))  # 2 means for 2d array
        self.fftw2_inverse = pyfftw.FFTW(array1, array2, axes=(0, 1), direction='FFTW_BACKWARD')
        self.fftw3 = pyfftw.FFTW(array3, array4, axes=(1, 2))
        self.fftw3_inverse = pyfftw.FFTW(array3, array4, axes=(1, 2), direction='FFTW_BACKWARD')
        self.fftw4 = pyfftw.FFTW(array5, array6, axes=(2, 3))
        self.fftw4_inverse = pyfftw.FFTW(array5, array6, axes=(2, 3), direction='FFTW_BACKWARD')
        ####pyfftw stuff####

        if parser.has_option('lens', 'atomFname'):
            atFname = parser.get('lens', 'atomFname')
            tmp = pyfits.getdata(atFname)
            '''Will consider revising this later--Shouzhuo'''
            tmp = np.fft.fftshift(tmp)
            nzl, nft, nyt, nxt = tmp.shape
            ypad = (self.ny - nyt) // 2
            xpad = (self.nx - nxt) // 2
            assert self.nframe == nft
            assert self.nzl == nzl
            ppad = ((0, 0), (0, 0), (ypad, ypad), (xpad, xpad))
            tmp = np.fft.ifftshift(np.pad(tmp, ppad))
            tmp = np.fft.fft2(tmp)
            self.fouaframesInter = tmp
            self.fouaframes = self.ks2D.transform(tmp, inFou=True, outFou=True)
            self.aframes = self.fftw2_inverse(self.fouaframes)
            # print('fouaframes shape: (line 751) ', self.fouaframes.shape)
        else:
            self.prepareFrames(parser)
        self.lensKernel = lensKernel

    def prepareFrames(self, parser):
        if parser.has_option('lens', 'SigmaFname'):
            self.__numerical_frames(parser)
            print("preparing numerical frames!!!!")
        else:
            self.__analytic_frames(parser)

    def __numerical_frames(self, parser):
        if parser.has_option('cosmology', 'omega_m'):
            omega_m = parser.getfloat('cosmology', 'omega_m')
        else:
            omega_m = Default_OmegaM
        self.cosmo = Cosmo(H0=Default_h0 * 100., Om0=omega_m)
        sigmaFname = parser.get('lens', 'SigmaFname')  # input fourier sigma field
        self.fouaframes = np.zeros(self.shapeA, dtype=np.complex128)
        self.aframes = np.zeros(self.shapeA, dtype=np.complex128)
        self.fouaframesInter = np.zeros(self.shapeA, dtype=np.complex128)
        unnormalized_sigma = fits.getdata(sigmaFname)
        for izl in range(self.nzl):
            '''no need to normalize'''
            # rpix = self.cosmo.angular_diameter_distance(self.zlBin[izl]).value / 180. * np.pi * self.scale
            # znorm = 1. / rpix ** 2.
            for ifr in reversed(range(self.nframe)):
                iAtomF = np.fft.fft2(np.fft.fftshift(unnormalized_sigma[izl, ifr]))  # fourier space sigma field
                # normTmp = iAtomF[0,0]/znorm
                iAtomF = iAtomF  # /normTmp
                self.fouaframesInter[izl, ifr] = iAtomF
                iAtomF = self.ks2D.transform(iAtomF, inFou=True, outFou=True)
                self.fouaframes[izl, ifr] = iAtomF
                self.aframes[izl, ifr] = self.fftw2_inverse(iAtomF)

    def __analytic_frames(self, parser):
        if parser.has_option('cosmology', 'omega_m'):
            omega_m = parser.getfloat('cosmology', 'omega_m')
        else:
            omega_m = Default_OmegaM
        self.cosmo = Cosmo(H0=Default_h0 * 100., Om0=omega_m)
        self.rs_base = parser.getfloat('lens', 'rs_base')  # Mpc/h
        self.resolve_lim = parser.getfloat('lens', 'resolve_lim')
        # Initialize basis predictors
        # In configure Space
        self.aframes = np.zeros(self.shapeA, dtype=np.complex128)
        # In Fourier space
        self.fouaframes = np.zeros(self.shapeA, dtype=np.complex128)
        # Intermediate basis in Fourier space
        self.fouaframesInter = np.zeros(self.shapeA, dtype=np.complex128)
        self.rs_frame = -1. * np.ones((self.nzl, self.nframe))  # Radius in pixel

        for izl in range(self.nzl):
            # the r_s for each redshift plane in units of pixel
            rpix = self.cosmo.angular_diameter_distance(self.zlBin[izl]).value / 180. * np.pi * self.scale
            # print('rpix is :',rpix)
            rz = self.rs_base / rpix
            # nfw halo with mass normalized to 1e14
            znorm = 1. / rpix ** 2.
            # angular scale of pixel size in Mpc
            for ifr in reversed(range(self.nframe)):
                # For each lens redshift bins, we begin from the
                # frame with largest angular scale radius
                rs = (ifr + 1) * rz  # older version that may not be a good sampling method
                rs = ifr * 0.05 / rpix + rz  # just chose a number that matches my reconstruction.
                # rs = ifr * 0.03 / rpix + rz
                # print('ifr',ifr)
                # print('rs',rs)
                if rs < self.resolve_lim:
                    # if one scale frame is less than resolution limit,
                    # skip this frame
                    break
                self.rs_frame[izl, ifr] = rs
                # print('rs is:', rs)
                # nfw halo with mass normalized to 1e14
                iAtomF = haloCS02SigmaAtom(r_s=rs, ny=self.ny, nx=self.nx, c=4., \
                                           sigma_pix=self.sigma_pix)
                normTmp = iAtomF[0, 0] / znorm
                iAtomF = iAtomF / normTmp
                self.fouaframesInter[izl, ifr] = iAtomF  # Fourier Space
                iAtomF = self.ks2D.transform(iAtomF, inFou=True, outFou=True)
                # KS transform
                self.fouaframes[izl, ifr] = iAtomF  # Fourier Space
                # self.aframes[izl,ifr]=np.fft.ifft2(iAtomF)  # Real Space
                self.aframes[izl, ifr] = self.fftw2_inverse(iAtomF)
                # print('iAtomF shape:', iAtomF.shape)
        return

    def itransformInter(self, dataIn):
        """
        transform from model (e.g., nfwlet) dictionary space to intermediate
        (e.g., delta) space
        """
        assert dataIn.shape == self.shapeA, \
            'input should have shape (nzl,nframe,ny,nx)'

        # convolve with atom in each frame/zlens (to Fourier space)
        # dataTmp =   np.fft.fft2(dataIn.astype(np.complex128),axes=(2,3))
        dataTmp = self.fftw4(dataIn)
        # print('dataIn shape: (itransformInter)', dataIn.astype(np.complex128).shape)
        dataTmp = dataTmp * self.fouaframesInter
        # sum over frames
        dataTmp = np.sum(dataTmp, axis=1)
        # back to configure space
        # dataOut =   np.fft.ifft2(dataTmp,axes=(1,2))
        dataOut = self.fftw3_inverse(dataTmp)
        # print('(itransformInter) dataTmp shape', dataTmp.shape)
        return dataOut

    def itransform(self, dataIn):
        """
        transform from model (e.g., nfwlet) dictionary space to measurement
        (e.g., shear) space
        Parameters:
            dataIn: array to be transformed (in configure space, e.g., alpha)
        """
        assert dataIn.shape == self.shapeA, \
            'input should have shape (nzl,nframe,ny,nx)'

        # convolve with atom in each frame/zlens (to Fourier space)
        # dataTmp =   np.fft.fft2(dataIn.astype(np.complex128),axes=(2,3))
        dataTmp = self.fftw4(dataIn)
        # print('dataIn shape (itransform):', dataIn.astype(np.complex128).shape)
        dataTmp = dataTmp * self.fouaframes
        # sum over frames
        dataTmp2 = np.sum(dataTmp, axis=1)
        # back to configure space
        # dataTmp2=   np.fft.ifft2(dataTmp2,axes=(1,2))
        dataTmp2 = self.fftw3_inverse(dataTmp2)
        # print('dataTmp2 shape (itransform):', dataTmp2.shape)
        # project to source plane
        dataOut = np.sum(dataTmp2[None, :, :, :] * self.lensKernel[:, :, None, None], axis=1)
        return dataOut

    def itranspose(self, dataIn):
        """
        transpose of the inverse transform operator
        Parameters:
            dataIn: arry to be operated (in config space, e.g., shear)
        """
        assert dataIn.shape == self.shapeS, \
            'input should have shape (nzs,ny,nx)'

        # Projection to lens plane
        # with shape=(nzl,nframe,ny,nx)
        dataTmp = np.sum(self.lensKernel[:, :, None, None] * dataIn[:, None, :, :], axis=0)
        # Convolve with atom*
        # dataTmp =   np.fft.fft2(dataTmp,axes=(1,2))
        dataTmp = self.fftw3(dataTmp)
        # print('dataIn shape (in itranspose):', dataTmp.shape)
        dataTmp = dataTmp[:, None, :, :] * np.conjugate(self.fouaframes)
        # The output with shape (nzl,nframe,ny,nx)
        # dataOut =   np.fft.ifft2(dataTmp,axes=(2,3))
        dataOut = self.fftw4_inverse(dataTmp)
        # print('dataTmp shape (in itranspose):', dataTmp.shape)
        return dataOut


def haloJS02SigmaAtom_mock_catalog(halo, scale, ny, nx, normalize=True, ra_0=0, dec_0=0):
    """
    Make a JS02 SigmaAtom. It seems the NFW counterpart, haloCS02SigmaAtom, takes in parameters of a halo and then outputs
    kappa field on a whole grid in fourier space. This is evident in
    Parameters:
        halo: just pass in a triaxial (or NFW) halo object
        ycgrid, xcgrid, the namesake of respective property in Grid(Cartesian) Object.
        the unit we are operating is arcmin.
    Note: whatever is being returned here is in configuration space.
    :param dec_0: offset (position of halo center)
    :param ra_0: offset (position of halo center)
    """
    Lx = nx * scale
    Ly = ny * scale
    nsamp = nx * ny * 2000  # better be 200  # making sure you have enough data points.
    ra = np.random.rand(nsamp) * Lx - Lx / 2. + ra_0
    dec = np.random.rand(nsamp) * Ly - Ly / 2 + dec_0
    # it seems the mass as normalized to be 1e14
    sigma_field = halo.Sigma(ra * 3600.,
                             dec * 3600.)  # just a Sigma field. This is also a 1d array, so you would have to pxielize it.
    if normalize:
        return sigma_field / (np.sum(sigma_field ** 2.)) ** 0.5, ra, dec, nsamp
    else:
        return sigma_field, ra, dec, nsamp


def haloJS02SigmaAtom_mock_catalog_dsigma(halo, scale, ny, nx, normalize=True, ra_0=0, dec_0=0):
    """
    Make a JS02 SigmaAtom. It seems the NFW counterpart, haloCS02SigmaAtom, takes in parameters of a halo and then outputs
    kappa field on a whole grid in fourier space. This is evident in
    Parameters:
        halo: just pass in a triaxial (or NFW) halo object
        ycgrid, xcgrid, the namesake of respective property in Grid(Cartesian) Object.
        the unit we are operating is arcmin.
    Note: whatever is being returned here is in configuration space.
    :param dec_0: offset (position of halo center)
    :param ra_0: offset (position of halo center)
    """
    Lx = nx * scale
    Ly = ny * scale
    nsamp = nx * ny * 2 #nx * ny * 2  # 2 to simulate realistic condition if nx*ny*10, reconstruction works better but too idealistic
    ra = np.random.rand(nsamp) * Lx - Lx / 2. + ra_0
    dec = np.random.rand(nsamp) * Ly - Ly / 2 + dec_0
    dsigma_field = halo.DeltaSigmaComplex(ra * 3600.,
                                          dec * 3600.)  # just a dSigma field. This is also a 1d array, so you would have to pxielize it.
    if normalize:
        return dsigma_field / (np.sum(np.abs(dsigma_field) ** 2.)) ** 0.5, ra, dec, nsamp  # I never use this anyways
    else:
        return dsigma_field, ra, dec, nsamp


class triaxialHalo(Cosmo):
    """
    Referencing mainly: Oguri, Lee, Suto 2003 (https://ui.adsabs.harvard.edu/abs/2003ApJ...599....7O/abstract)
    Hereafter OLS03
    and Jing, Suto 2002 (https://ui.adsabs.harvard.edu/abs/2002ApJ...574..538J/abstract)
    Hereafter JS02

    Parameters:
        mass:       Mass defined using a spherical overdensity of 200 times the
                    critical density of the universe, in units of M_solar/h.
        conc:       Halo concentration parameter, Re / R0. This is ce defined in eqn 5 of OLS03
        redshift:   Redshift of the halo.
        ra:         ra of halo center  [arcsec].
        dec:        dec of halo center [arcsec].
        omega_m:    Omega_matter to pass to Cosmology constructor, omega_l is
                    set to 1-omega_matter. (default: Default_OmegaM)
        a_over_b:   look at eqn 4 of OLS03. This is ratio of the length of the axis in the density distribution
        a_over_c:   same as above, but this number should be smaller than a_over_b, because a<=b<=c

        IMPORTANT: SINCE IT IS SEEN THAT ALPHA=1.5 REPRODUCES ARC STAT CORRECTLY, WE DO THINGS WITH ALPHA = 1.5
        tri_nfw:    if the halo is a triaxial version of NFW halos (alpha=1)
        OLS03:   Use original definition of OLS03, which involves ce and Re extensively.
    """

    def __init__(self, ra, dec, redshift, mass, a_over_b, a_over_c, conc, phi_prime=0, theta_prime=0, rs=None,
                 omega_m=Default_OmegaM, tri_nfw=False, OLS03=False):
        # Redshift and Geometry
        # ra dec
        # self in here seems to be the Cosmo object
        self.OLS03 = OLS03
        self.ra = ra  # right ascension in celestial coordinates
        self.dec = dec  # declination in celestial coordinates
        self.a_over_b = a_over_b  # not if should have
        self.a_over_c = a_over_c  # not_if_should_have
        self.c = np.float128(conc)  # the concentration parameter, this virial radius over R0.
        self.ce = self.c * 0.45
        self.phi_prime = phi_prime  # the intrinsic halo alignment with respect to earth's coordinate. In radians
        self.theta_prime = theta_prime  # the intrisic halo aignment with respect to earth's coordinate. In radians
        Cosmo.__init__(self, H0=Default_h0 * 100., Om0=omega_m)
        self.z = float(redshift)
        self.a = 1. / (1. + self.z)  # scale factor

        # angular distance in Mpc/h
        self.DaLens = self.angular_diameter_distance_z1z2(0., self.z).value

        # critical density of the universe
        # in unit of M_solar / Mpc^3
        rho_cZ = self.critical_density(self.z).to_value(unit=rho_unt)
        self.rho_cZ = rho_cZ
        self.M = np.float128(mass)

        # First, we get the virial radius, which is defined for some spherical
        # overdensity as 3 M / [4 pi (r_vir)^3] = overdensity. Here we have
        # overdensity = 200 * rhocrit, to determine r_vir (angular distance).
        # The factor of 1.63e-5 [h^(-2/3.)] comes from the following set of prefactors:
        # (3 / (4 pi * 200 * rhocrit))^(1/3), where rhocrit = 2.8e11 h^2
        # M_solar / Mpc^3.
        # (DH=C_LIGHT/1e3/100/h,rho_crit=1.5/four_pi_G_over_c_squared()/(DH)**2.)
        ezInv = self.inv_efunc(redshift)  # This is 1/H(z). property of astropy flat lambdaCDM.
        self.ezInv = ezInv
        self.Omega_z = self.Om(self.z)  # matter (non-relativistic) density parameter, after eqn 6 of OLS03
        # self.Delta_vir = 18 * np.pi ** 2 + 82 * (self.Omega_z - 1) - 39 * (
        #         self.Omega_z - 1) ** 2  # Used eqn 6 OLS03, eqn 1 JS02
        self.omega_vir = 1 / (self.Om0 * (1 + self.z) ** 3 / (self.Om0 * (1 + self.z) ** 3 + 1 - self.Om0)) - 1
        self.Delta_vir = 18 * np.pi ** 2 * (1 + 0.4093 * self.omega_vir ** (0.9052))  # Used eqn 6 OLS03, see Oguri,
        # Taruya, Suto 2001, 559, 572-583 eqn 4.

        # self.rvir = self.M ** (1 / 3) * (3 / np.pi) ** (1 / 3) / (
        #         2 ** (2 / 3) * self.Delta_vir ** (1 / 3) * rho_cZ ** (1 / 3))
        self.rvir = (3 * self.M / (4 * np.pi * self.Delta_vir * self.Omega_z * self.rho_cZ)) ** (1 / 3)

        ### Older Version
        self.Re = 0.45 * self.rvir  # at top of the page 9 on OLS03. An empirical/// relation between rvir and Re.
        self.R0 = 0.45 * self.rvir / self.ce  # equation 10 of OLS03.
        self.rs = self.R0
        if self.OLS03:
            self.Delta_e = 5 * self.Delta_vir * (self.a_over_b / self.a_over_c / self.a_over_c) ** 0.75
        else:
            self.Delta_e = self.Delta_vir * (self.a_over_b / self.a_over_c / self.a_over_c) ** 0.75  # eqn 6 OLS03
        # is 5 * self.Delta_vir * (self.a_over_b / self.a_over_c / self.a_over_c) ** 0.75, but this way integrating the mass
        # to virial radius does not give the correct mass.
        ### Older Version
        if self.OLS03:
            if tri_nfw:
                self.m_c = np.log(1 + self.ce) - self.ce / (1 + self.ce)
            else:
                self.m_c = (2 * np.log(np.sqrt(self.ce) + np.sqrt(1 + self.ce)) - 2 * np.sqrt(
                    self.ce / (1 + self.ce)))
        else:
            if tri_nfw:
                self.m_c = np.log(1 + self.c) - self.c / (1 + self.c)
            else:
                self.m_c = (2 * np.log(np.sqrt(self.c) + np.sqrt(1 + self.c)) - 2 * np.sqrt(
                    self.c / (1 + self.c)))  # eqn 9 OLS03, alpha = 1.5. I believe a typo as been made between c and ce.

        # self.m_c =  self.c ** (3 - 1) / 2 * (np.log(1+self.c) - self.c/(1+self.c))  # alpha = 1
        if self.OLS03:
            self.delta_triaxial = self.Delta_e * self.Omega_z / 3. * self.ce ** 3. / self.m_c
        else:
            self.delta_triaxial = self.Delta_e * self.Omega_z / 3. * self.c ** 3. / self.m_c  # eqn 7 OLS 03
        # convert to angular radius in unit of arcsec
        scale = self.R0 / self.DaLens
        arcsec2rad = np.pi / 180. / 3600.
        self.rs_arcsec = scale / arcsec2rad
        self.rvir_arcsec = (self.rvir / self.DaLens) / arcsec2rad

        # Second, derive the charateristic matter density
        # within virial radius at redshift z
        self.rho_s = rho_cZ * self.delta_triaxial  # see eqn 3 OLS03
        if a_over_b > 1:
            raise ValueError("a should smaller than or equal to b")
        if a_over_c > 1:
            raise ValueError("a should smaller than or equal to c")
        if a_over_b < a_over_c:
            raise ValueError("b should smaller than or equal to c")
        return

    def DdRs(self, ra_s, dec_s):
        """Calculate 'x' the radius r in units of the Triaxial scale
        radius, r_s.
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        x = ((ra_s - self.ra) ** 2 + (dec_s - self.dec) ** 2) ** 0.5 / self.rs_arcsec
        return x

    def DdRvir(self, ra_s, dec_s):
        """Calculate 'x' the radius r in units of the Triaxial scale
        radius, r_s.
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        x = ((ra_s - self.ra) ** 2 + (dec_s - self.dec) ** 2) ** 0.5 / self.rvir_arcsec
        return x

    def sin2phi(self, ra_s, dec_s):
        """Calculate cos2phi and sin2phi with reference to the halo center
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # pure tangential shear, no cross component
        dx = ra_s - self.ra
        dy = dec_s - self.dec
        drsq = dx * dx + dy * dy
        return np.divide(2 * dx * dy, drsq, where=(drsq != 0.))

    def cos2phi(self, ra_s, dec_s):
        """Calculate cos2phi and sin2phi with reference to the halo center
        Parameters:
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """
        # pure tangential shear, no cross component
        dx = ra_s - self.ra
        dy = dec_s - self.dec
        drsq = dx * dx + dy * dy
        return np.divide(dx * dx - dy * dy, drsq, where=(drsq != 0.))

    def lensKernel(self, z_s):
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
        k_s = np.zeros(len(z_s))
        mask = z_s > self.z
        k_s[mask] = self.angular_diameter_distance_z1z2(self.z, z_s[mask]) * self.DaLens \
                    / self.angular_diameter_distance_z1z2(0., z_s[mask]) * four_pi_G_over_c_squared()
        return k_s


class triaxialJS02(triaxialHalo):
    """
    Integral functions of a triaxial density profile:
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
        tri_nfw:    if set to true, returns sigma and dsigma associate with alpha=1 (triaxial version of nfw halo)
        long_truncation:
                    if set to true, enforce a cutoff at x = 2c instead of x = c.
    """

    def __init__(self, ra, dec, redshift, mass, a_over_b, a_over_c, conc=None, phi_prime=0, theta_prime=0, rs=None,
                 omega_m=Default_OmegaM, tri_nfw=False, long_truncation=False, OLS03=False):
        triaxialHalo.__init__(self, ra, dec, redshift, mass, a_over_b, a_over_c, conc, phi_prime=phi_prime,
                              theta_prime=theta_prime, rs=None,
                              omega_m=omega_m, tri_nfw=tri_nfw, OLS03=OLS03)
        self.c2_a2 = self.a_over_c ** (-2)
        self.c2_b2 = (np.abs(self.a_over_c / self.a_over_b)) ** (-2)
        self.phi = self.cal_phi()
        self.theta = self.cal_theta()
        self.f = self.f(self.theta, self.phi)
        self.b_TNFW = self.b_TNFW_over_sigma_crit(self.f)
        self.A = self.A(self.theta, self.phi)
        self.B = self.B(self.theta, self.phi)
        self.C = self.C(self.theta, self.phi)
        self.qx = self.qx(self.f, self.A, self.B, self.C)  ####qx, NOT QX SQUARED!!!
        self.qy = self.qy(self.f, self.A, self.B, self.C)  #### qy NOT QY SQUARED!!!
        self.q = self.q(self.qx, self.qy)
        self.tri_nfw = tri_nfw
        if np.abs(self.B) < 1e-8:
            self.psi = 0
        else:
            self.psi = 1 / 2 * np.arctan(self.B / (self.A - self.C))
        self.u_array = np.logspace(np.log10(1e-8), 0, 1000)
        # self.u_array = np.logspace(np.log10(1e-8), 0, 2000)
        self.u_val = np.array([])
        for i in range(len(self.u_array) - 1):
            self.u_val = np.append(self.u_val, (self.u_array[i + 1] + self.u_array[i]) / 2)
        self.dlogu = np.log(self.u_array[1] / self.u_array[0])
        self.long_truncation = long_truncation

    def f_GNFW(self, r):
        ''' Eqn 41 in OLS. The input r takes zeta, defined in 21 of OLS'''
        if not self.tri_nfw:
            # truncated alpha=1.5 case. NOTE THIS ONLY APPLIES TO C=4!!!!!!
            # f_GNFW_val = 4.9257573299999995 * 10 ** (-6) * r ** 0.0422695641 + 8.87596287 / (
            #             5.08395682 * r ** 0.557570911 + 14.4920247 * r ** 1.54971972 +
            #             2.0591881 * r ** 3.03259341)
            # for untruncated values
            top = 2.614
            bottom = r ** 0.5 * (1 + 2.378 * r ** 0.5833 + 2.617 * r ** (3 / 2))
            f_GNFW_val = np.divide(top, bottom)
        else:
            # this has potential to have any different concentration parameter.
            x = r
            f_GNFW_val = np.zeros_like(r, dtype=np.float128)
            mask = np.where(x < 0.999)[0]
            a = ((1 - x[mask]) / (x[mask] + 1)) ** 0.5
            f_GNFW_val[mask] = 1 / (x[mask] ** 2 - 1) * (1 - 2 * np.arctanh(a) / (1 - x[mask] ** 2) ** 0.5)

            mask = np.where(x > 1.001)[0]
            a = ((x[mask] - 1) / (x[mask] + 1)) ** 0.5
            f_GNFW_val[mask] = 1 / (x[mask] ** 2 - 1) * (1 - 2 * np.arctan(a) / (x[mask] ** 2 - 1) ** 0.5)

            # the approximation below has a maximum fractional error of 7.4e-7
            mask = np.where((x >= 0.999) & (x <= 1.001))[0]
            f_GNFW_val[mask] = (22. / 15. - 0.8 * x[mask]) / 2
            if self.long_truncation:
                # mask = np.where(x > 2 * self.c)[0]
                mask = np.where(x > np.inf)[0]  # no cut-off
            else:
                mask = np.where(x > self.c)[0]
            f_GNFW_val[mask] = 0
            '''c = float(self.c)
            x0 = r
            out = np.zeros_like(x0, dtype=float)

            # 3 cases: x < 1-0.001, x > 1+0.001, and |x-1| < 0.001
            mask = np.where(x0 < 0.999)[0]
            x = x0[mask]
            out[mask] = -np.sqrt(c ** 2. - x ** 2.) / (1 - x ** 2.) / (1 + c) + \
                        1. / (1 - x ** 2.) ** 1.5 * np.arccosh((x ** 2. + c) / x / (1. + c))

            mask = np.where((x0 > 1.001) & (x0 < c))[0]
            x = x0[mask]
            out[mask] = -np.sqrt(c ** 2. - x ** 2.) / (1 - x ** 2.) / (1 + c) - \
                        1. / (x ** 2. - 1) ** 1.5 * np.arccos((x ** 2. + c) / x / (1. + c))

            mask = np.where((x0 >= 0.999) & (x0 <= 1.001))[0]
            x = x0[mask]
            out[mask] = (-2. + c + c ** 2.) / (3. * np.sqrt(-1. + c) * (1 + c) ** (3. / 2)) \
                        + ((2. - c - 4. * c ** 2. - 2. * c ** 3.) * (x - 1.)) / (
                                5. * np.sqrt(-1. + c) * (1 + c) ** (5 / 2.))

            mask = np.where(x0 >= c)[0]
            out[mask] = 0.
            f_GNFW_val = out'''
            # Below are untruncated triaxial profiles. three cases x<1, x>1, and |x-1|<0.001

            # top = np.ones_like(r, dtype=np.float128)
            # bottom = np.ones_like(r, dtype=np.float128)
            # mask = np.where(r < 0.999)[0]
            # top[mask] = -1 + 2 / (np.sqrt(1 - r[mask] ** 2)) * np.arctanh(np.sqrt((1 - r[mask]) / (1 + r[mask])))
            # bottom[mask] = 1 - r[mask] ** 2
            # mask = np.where((r >= 0.999) & (r <= 1.001))[0]
            # top[mask] = 1 / 3
            # bottom[mask] = 1
            # mask = np.where(r > 1.001)[0]
            # top[mask] = 1 - 2 / (np.sqrt(r[mask] ** 2 - 1)) * np.arctan(np.sqrt((r[mask] - 1) / (1 + r[mask])))
            # bottom[mask] = -1 + r[mask] ** 2

        return f_GNFW_val

    def f_GNFW_xi(self, xi):
        if not self.tri_nfw:
            # for truncated alpha=1.5 case. NOTE THIS ONLY APPLIES TO C=4!!!!
            top = 2.614
            qx = self.qx
            bottom = (xi / qx) ** 0.5 * (1 + 2.378 * (xi / qx) ** 0.5833 + 2.617 * (xi / qx) ** (3 / 2))
            f_GNFW_val = np.divide(top, bottom)
            # f_GNFW_val = 4.9257573299999995 * 10 ** (-6) * (xi / qx) ** 0.0422695641
            # + 8.87596287 / (5.08395682 * (xi / qx) ** 0.557570911 + 14.4920247 * (xi / qx) ** 1.54971972 +
            #                 2.0591881 * (xi / qx) ** 3.03259341)
        else:
            # alpha = 1, untruncated
            x = xi / self.qx
            f_GNFW_val = np.zeros_like(xi, dtype=np.float128)
            mask = np.where(x < 0.999)[0]
            a = ((1 - x[mask]) / (x[mask] + 1)) ** 0.5
            f_GNFW_val[mask] = 1 / (x[mask] ** 2 - 1) * (1 - 2 * np.arctanh(a) / (1 - x[mask] ** 2) ** 0.5)

            mask = np.where(x > 1.001)[0]
            a = ((x[mask] - 1) / (x[mask] + 1)) ** 0.5
            f_GNFW_val[mask] = 1 / (x[mask] ** 2 - 1) * (1 - 2 * np.arctan(a) / (x[mask] ** 2 - 1) ** 0.5)

            # the approximation below has a maximum fractional error of 7.4e-7
            mask = np.where((x >= 0.999) & (x <= 1.001))[0]
            f_GNFW_val[mask] = (22. / 15. - 0.8 * x[mask]) / 2
            if self.long_truncation:
                mask = np.where(x > np.inf)[0]
            else:
                mask = np.where(x > self.c)[0]
            f_GNFW_val[mask] = 0
        return f_GNFW_val

    def b_TNFW_over_sigma_crit(self, f):
        '''eqn 24 in OLS, without the sigma_crit, because we need to multiply that to the expressionwhen we find
        density anyways '''
        return 4 * self.rho_s * self.R0 * 1 / np.sqrt(f)

    def cal_xp(self, ra_s, dec_s):
        '''We divide by rs_arcsec because every x,y,z is scaled by R0'''
        x = (ra_s - self.ra) / self.rs_arcsec  # normalized (-) xp in Fig 1 OLS. We are cool b/c sign doesn't matter
        return x

    def cal_yp(self, ra_s, dec_s):
        y = (
                    dec_s - self.dec) / self.rs_arcsec  # normalized (-) yp in Fig 1 OLS. We are cool b/c sign doesn't matter (AND ELLIPSOID SYMMETRY)
        return y

    def zeta(self, h, g, f):
        '''Eqn 21, OLS Typo in the expression. It should be zeta^2 on the LHS'''
        return np.sqrt(np.abs(h - g ** 2 / (4 * f)))

    def f(self, theta, phi):
        '''eqn 17 in OLS'''
        c2_a2 = self.c2_a2
        c2_b2 = self.c2_b2
        return np.sin(theta) ** 2 * (c2_a2 * np.cos(phi) ** 2 + c2_b2 * np.sin(phi) ** 2) + np.cos(theta) ** 2

    def g(self, theta, phi, xp, yp):
        '''eqn 18, OLS'''
        c2_a2 = self.c2_a2
        c2_b2 = self.c2_b2
        firstline = np.sin(theta) * np.sin(2 * phi) * (c2_b2 - c2_a2) * xp
        secondline = np.sin(2 * theta) * (1 - c2_a2 * np.cos(phi) ** 2 - c2_b2 * np.sin(phi) ** 2) * yp
        return firstline + secondline

    def h_ols(self, theta, phi, xp, yp):
        '''eqn 19, OLS'''
        c2_a2 = self.c2_a2
        c2_b2 = self.c2_b2
        firstline = (c2_a2 * np.sin(phi) ** 2 + c2_b2 * np.cos(phi) ** 2) * xp ** 2 + np.sin(2 * phi) * np.cos(
            theta) * (c2_a2 - c2_b2) * xp * yp
        secondline = (np.cos(theta) ** 2 * (c2_a2 * np.cos(phi) ** 2 + c2_b2 * np.sin(phi) ** 2) + np.sin(
            theta) ** 2) * yp ** 2
        return firstline + secondline

    def cal_phi(self):
        '''ra_s : right ascension of line of sight, which I take as the source?
        dec: declination of line of sight. Fig 1 of OLS'''
        arcsec2rad = np.pi / 180. / 3600
        phi = self.ra * arcsec2rad
        return phi - self.phi_prime

    def cal_theta(self):
        '''ra_s : right ascension of line of sight, which I take as the source?
                dec: declination of line of sight
                refer to euler angle or Fig 1 of OLS'''
        arcsec2rad = np.pi / 180. / 3600.
        theta = np.pi / 2. - self.dec * arcsec2rad
        # after considering intrinsic alignment of halo
        return theta - self.theta_prime

    def __Sigma(self, x0, ra_s0, dec_s0):
        c = np.float128(self.c)
        out = np.zeros_like(x0, dtype=float)
        if self.long_truncation:
            # mask = np.where((x0 >= 0) & (x0 <= 2 * c))[0]
            mask = np.where((x0 >= 0) & (x0 <= np.inf))  # not cutoff
        else:
            mask = np.where((x0 >= 0) & (x0 <= c))[0]
        x = x0[mask]
        ra_s = ra_s0[mask]
        dec_s = dec_s0[mask]
        theta = self.theta
        phi = self.phi
        f = self.f
        xp = self.cal_xp(ra_s, dec_s)
        yp = self.cal_yp(ra_s, dec_s)
        g = self.g(theta, phi, xp, yp)
        h = self.h_ols(theta, phi, xp, yp)
        zeta = self.zeta(h, g, f)
        bTNFW = self.b_TNFW_over_sigma_crit(f)
        f_GNFW = self.f_GNFW(zeta)
        out[mask] = bTNFW * f_GNFW / 2
        # the undefined are just zero
        return out

    def Sigma(self, ra_s, dec_s):
        """Calculate Surface Density (Sigma) of halo.
        Takada & Jain(2003, MNRAS, 340, 580) Eq.27
            ra_s:       ra of sources [arcsec].
            dec_s:      dec of sources [arcsec].
        """

        # convenience: call with single number
        assert isinstance(ra_s, np.ndarray) == isinstance(dec_s, np.ndarray), \
            'ra_s and dec_s do not have same type'
        if not isinstance(ra_s, np.ndarray):
            ra_sA = np.array([ra_s], dtype='float')
            dec_sA = np.array([dec_s], dtype='float')
            return self.Sigma(ra_sA, dec_sA)[0]
        assert len(ra_s) == len(dec_s), \
            'input ra and dec have different length '
        x = self.DdRs(ra_s, dec_s)
        return self.__Sigma(x, ra_s, dec_s)

    def qx(self, f, A, B, C):
        '''A2 in CK'''
        if A >= C:
            qx2 = 2 * f / (A + C + np.sqrt((A - C) ** 2 + B ** 2))
        else:
            qx2 = 2 * f / (A + C - np.sqrt((A - C) ** 2 + B ** 2))
        # when C>A, the positions should be swtiched
        return np.sqrt(np.abs(qx2))

    def qy(self, f, A, B, C):
        '''A3 in CK'''
        if A >= C:
            qy2 = 2 * f / (A + C - np.sqrt((A - C) ** 2 + B ** 2))
        else:
            qy2 = 2 * f / (A + C + np.sqrt((A - C) ** 2 + B ** 2))
        # when C>A, the positions should be swtiched
        return np.sqrt(np.abs(qy2))

    def q(self, qx, qy):
        return qy / qx

    def A(self, theta, phi):
        '''A5 in CK'''
        c2_a2 = self.a_over_c ** (-2)
        c2_b2 = (np.abs(self.a_over_c / self.a_over_b)) ** (-2)
        A = np.cos(theta) ** 2 * (c2_a2 * np.sin(phi) ** 2 + c2_b2 * np.cos(phi) ** 2) + c2_a2 * c2_b2 * np.sin(
            theta) ** 2
        # print("A is", A)
        # print("c2b2", c2_b2)
        # print("phi is", phi)
        return A

    def B(self, theta, phi):
        '''A6 in CK'''
        c2_a2 = self.a_over_c ** (-2)
        c2_b2 = (np.abs(self.a_over_c / self.a_over_b)) ** (-2)
        B = np.cos(theta) * np.sin(2 * phi) * (c2_a2 - c2_b2)
        # print("B is", B)
        return B

    def C(self, theta, phi):
        '''A7 in CK'''
        c2_a2 = self.a_over_c ** (-2)
        c2_b2 = (np.abs(self.a_over_c / self.a_over_b)) ** (-2)
        return c2_b2 * np.sin(phi) ** 2 + c2_a2 * np.cos(phi) ** 2

    def zeta2(self, u, xp, yp, qx, q):
        '''unlike the other zeta, this takes different arguments. This also returns ZETA SQUARED!!!
        A18 in CK
        Most likely gonna be obsolete.'''
        # before, qx is not squared. I believe if u = 1 in A18, you should be able to get A1, which you don't if qx is not squared.
        return u / (qx ** 2) * (xp ** 2 + yp ** 2 / (1 - (1 - q ** 2) * u))

    def xi2(self, u, xpp, ypp):
        '''xi defined right after eqn 35 in OLS03. This returns xi SQUARED!!!!. Eqn 39 of OLS03'''
        q = self.q
        return u * (xpp ** 2 + ypp ** 2 / (1 - (1 - q ** 2) * u))

    def kappa_without_factor(self, zeta):
        '''23 in OLS... again, but but with different input options. Also notice that I didn't include Sigma_crit here.
        THE PROGRAM WILL GIVE THE CORRECT SHEAR AFTER BEING MULTIPLIED BY LENSING KERNEL'''
        f = self.f
        btnfw = self.b_TNFW_over_sigma_crit(f)
        fGNFW = self.f_GNFW(zeta)
        # return btnfw * fGNFW / 2
        return fGNFW

    def sigma_multipleways(self, ra_s, dec_s):
        '''Use different eqn for zeta'''
        xp = self.cal_xp(ra_s, dec_s)
        yp = self.cal_yp(ra_s, dec_s)
        zeta = np.sqrt((self.A * xp ** 2 + self.B * xp * yp + self.C * yp * yp) / self.f)
        xpp, ypp = self.prime_to_double_prime(xp, yp, self.A, self.B, self.C)
        xi = np.sqrt(self.xi2(1, xpp, ypp))
        zeta_another = np.sqrt(xpp ** 2 / self.qx ** 2 + ypp ** 2 / self.qy ** 2)
        h = self.h_ols(self.theta, self.phi, xp, yp)
        g = self.g(self.theta, self.phi, xp, yp)
        return self.kappa_without_factor(zeta) * self.b_TNFW / 2, self.f_GNFW_xi(xi) * self.b_TNFW / 2, \
               self.kappa_without_factor(zeta_another) * self.b_TNFW / 2, zeta_another / zeta

    def kappa_prime(self, zeta_0, theta, phi):
        '''kappa prime at some point, needed for later. zeta_0 is the place you want to take derivative with respect to'''
        d_zeta = 1e-6
        return derivative(self.kappa_without_factor, zeta_0, dx=d_zeta)

    def f_GNFW_xi_prime_numerical(self, xi_0):
        d_xi = 1e-6
        return derivative(self.f_GNFW_xi, xi_0, dx=d_xi, )

    def f_gnfw_prime_compare(self, r):
        '''Comparison between analytic and numerical differentiation of f with respect to xi'''
        d_r = 0.00001
        zeta = r
        qx = 1.7
        xi = zeta / qx
        first_top = - 2.614 * (1.38709 / (qx * (xi / qx) ** 0.4167) + 3.9255 * (xi / qx) ** 0.5 / qx)
        first_bottom = (xi / qx) ** 0.5 * (1 + 2.378 * (xi / qx) ** 0.5833 + 2.617 * (xi / qx) ** 1.5) ** 2
        second_top = -1.307
        second_bototm = qx * (xi / qx) ** 1.5 * (1 + 2.378 * (xi / qx) ** 0.5833 + 2.617 * (xi / qx) ** 1.5)
        return derivative(self.f_GNFW_xi, r, dx=d_r, args=(qx,)) / qx - (
                first_top / first_bottom + second_top / second_bototm)

    def f_GNFW_xi_prime_analytic(self, xi):
        '''Taking derivative with respect to xi in OLS 041.'''
        qx = self.qx
        if not self.tri_nfw:
            qx = self.qx
            first_top = - 2.614 * (1.38709 / (qx * (xi / qx) ** 0.4167) + 3.9255 * (xi / qx) ** 0.5 / qx)
            first_bottom = (xi / qx) ** 0.5 * (1 + 2.378 * (xi / qx) ** 0.5833 + 2.617 * (xi / qx) ** 1.5) ** 2
            second_top = -1.307
            second_bototm = qx * (xi / qx) ** 1.5 * (1 + 2.378 * (xi / qx) ** 0.5833 + 2.617 * (xi / qx) ** 1.5)
            out = first_top / first_bottom + second_top / second_bototm
            # truncated alpha = 1.5 case. ONLY WORKS WITH C=4
            # return 2.0820961520147983e-7/(qx*(xi/qx)**0.9577304359) -(8.87596287*(2.834666435612063/(qx*(xi/qx)**0.442429089) +
            # (22.458576460317083*(xi/qx)**0.5497197199999999)/
            # qx + (6.244680262010421*(xi/qx)**2.03259341)/qx))/(5.08395682*(xi/qx)**0.557570911 + 14.4920247*(xi/qx)**1.54971972 + 2.0591881*(xi/qx)**3.03259341)**2
        else:
            # alpha =1 untruncated.
            x = xi / self.qx
            qx = self.qx
            out = np.zeros_like(xi, dtype=np.float128)
            mask = np.where(x < 0.999)[0]
            out[mask] = -((qx ** 2 * (2 * xi[mask] ** 2 * np.sqrt(1 - xi[mask] ** 2 / qx ** 2) + qx ** 2 * np.sqrt(
                -1 + (2 * qx) / (qx + xi[mask])) + qx * xi[mask] * np.sqrt(-1 + (2 * qx) / (qx + xi[mask])) - 6 * xi[
                                          mask] ** 2 * np.arctanh(np.sqrt(-1 + (2 * qx) / (qx + xi[mask]))))) / (
                                  xi[mask] * (qx ** 2 - xi[mask] ** 2) ** 2 * np.sqrt(1 - xi[mask] ** 2 / qx ** 2)))

            mask = np.where(x > 1.001)[0]
            out[mask] = (qx ** 2 * (
                    qx ** 2 - qx * xi[mask] - 2 * xi[mask] ** 2 * np.sqrt(-1 + xi[mask] ** 2 / qx ** 2) * np.sqrt(
                1 - (2 * qx) / (qx + xi[mask])) + 6 * xi[mask] ** 2 * np.sqrt(
                1 - (2 * qx) / (qx + xi[mask])) * np.arctan(np.sqrt(1 - (2 * qx) / (qx + xi[mask]))))) / (
                                xi[mask] * (qx ** 2 - xi[mask] ** 2) ** 2 * np.sqrt(
                            -1 + xi[mask] ** 2 / qx ** 2) * np.sqrt(1 - (2 * qx) / (qx + xi[mask])))
            # the approximation below has a maximum fractional error of 7.4e-7
            mask = np.where((x >= 0.999) & (x <= 1.001))[0]
            out[mask] = - 0.4 * x[mask]
        return out

    def K_n_integrand(self, u, xpp, ypp, n):
        '''A16 in Ck. From Keeton  https://arxiv.org/pdf/astro-ph/0102341.pdf, the derivative should be about xi.
        The 1/2 /xi involves keeping with the notation of phixx phi yy and the chain rule being with respect to xi instead xi^2
        '''
        xi = np.sqrt(np.abs(self.xi2(u, xpp, ypp)))
        f_prime = self.f_GNFW_xi_prime_analytic(xi)
        # f_prime = self.f_GNFW_xi_prime_numerical(xi)
        return u * f_prime / ((1 - (1 - self.q ** 2) * u) ** (n + 1 / 2)) * 1 / 2 / xi
        # return u * f_prime / ((1 - (1 - self.q ** 2) * u) ** (n + 1 / 2))

    def J_n_integrand(self, u, xpp, ypp, n):
        '''A17 in Ck'''
        xi = np.sqrt(np.abs(self.xi2(u, xpp, ypp)))
        f_GNFW = self.f_GNFW_xi(xi)
        return f_GNFW / ((1 - (1 - self.q ** 2) * u) ** (n + 1 / 2))

    def prime_to_double_prime(self, xp, yp, A, B, C):
        '''Goes from x',y' in eqn 21 to x'', y'' in equation 32 in OLS.
        There's just a rotation, but in the oppositive direction.
        It is better to verify this numerically rather than analyticially.'''
        psi = self.psi  # 1 / 2 * np.arctan(B / (A - C))
        xpp = xp * np.cos(psi) + yp * np.sin(psi)
        ypp = yp * np.cos(psi) - xp * np.sin(psi)
        ##### if A>C, it would seem that qx, qy are switched. But if you go ahead to calculate the exact value of
        ##### zeta, in order for it to stay the same (preserve the quadratic form), you don't have to switch them.
        return xpp, ypp

    def K_n(self, ra_s, dec_s, n):
        '''A16 in CK. I transformed to double prime realm here.'''
        xp = np.ravel(self.cal_xp(ra_s, dec_s))
        yp = np.ravel(self.cal_yp(ra_s, dec_s))
        A = self.A
        B = self.B
        C = self.C
        xpp, ypp = self.prime_to_double_prime(xp, yp, A, B, C)
        length = len(xp)
        out = np.zeros(length, dtype=float)
        for i in range(length):
            out[i] = quad(self.K_n_integrand, 0, 1, args=(xpp[i], ypp[i], n))[0]
        return out * self.b_TNFW / 2

    def J_n(self, ra_s, dec_s, n):
        '''A17 in CK'''
        xp = np.ravel(self.cal_xp(ra_s, dec_s))
        yp = np.ravel(self.cal_yp(ra_s, dec_s))
        A = self.A
        B = self.B
        C = self.C
        xpp, ypp = self.prime_to_double_prime(xp, yp, A, B, C)
        length = len(xp)
        out = np.zeros(length, dtype=float)
        for i in range(length):
            out[i] = quad(self.J_n_integrand, 0, 1, args=(xpp[i], ypp[i], n))[0]
        return out * self.b_TNFW / 2  # * 1/np.sqrt(self.f) ##actuall b_TNFW / lensing kernel

    def log_integral_Jn(self, xpp, ypp, n):
        '''A regime for logrithmically integrate. Note that this only applies to Jn, Kn
        Also note only 1 value should be brought into this function because the integrand
        function cannot handle inputs with various dimensions'''
        return np.sum(self.J_n_integrand(self.u_val, xpp, ypp, n) * self.u_val) * self.dlogu

    def log_integral_Kn(self, xpp, ypp, n):
        '''A regime for logrithmically integrate. Note that this only applies to Jn, Kn'''
        return np.sum(self.K_n_integrand(self.u_val, xpp, ypp, n) * self.u_val) * self.dlogu

    def K_n_fast(self, ra_s, dec_s, n):
        xp = np.ravel(self.cal_xp(ra_s, dec_s))
        yp = np.ravel(self.cal_yp(ra_s, dec_s))
        A = self.A
        B = self.B
        C = self.C
        xpp, ypp = self.prime_to_double_prime(xp, yp, A, B, C)
        length = len(xp)
        out = np.zeros(length, dtype=float)
        for i in range(length):
            out[i] = self.log_integral_Kn(xpp[i], ypp[i], n)
        return out * self.b_TNFW / 2

    def J_n_fast(self, ra_s, dec_s, n):
        xp = np.ravel(self.cal_xp(ra_s, dec_s))
        yp = np.ravel(self.cal_yp(ra_s, dec_s))
        A = self.A
        B = self.B
        C = self.C
        xpp, ypp = self.prime_to_double_prime(xp, yp, A, B, C)
        length = len(xp)
        out = np.zeros(length, dtype=float)
        for i in range(length):
            out[i] = self.log_integral_Jn(xpp[i], ypp[i], n)
        return out * self.b_TNFW / 2

    def phi_xx(self, ra_s, dec_s, fast=True):
        '''A13 in CK'''
        q = self.q
        xp = self.cal_xp(ra_s, dec_s)
        yp = self.cal_xp(ra_s, dec_s)
        xpp, ypp = self.prime_to_double_prime(xp, yp, self.A, self.B, self.C)
        if fast:
            K_0 = self.K_n_fast(ra_s, dec_s, 0)
            J_0 = self.J_n_fast(ra_s, dec_s, 0)
        else:
            K_0 = self.K_n(ra_s, dec_s, 0)
            J_0 = self.J_n(ra_s, dec_s, 0)
        K_0 = np.reshape(K_0, np.shape(ra_s))
        J_0 = np.reshape(J_0, np.shape(ra_s))
        return 2 * q * xpp ** 2 * K_0 + q * J_0

    def phi_yy(self, ra_s, dec_s, fast=True):
        '''A14 in CK'''
        q = self.q
        xp = self.cal_xp(ra_s, dec_s)
        yp = self.cal_yp(ra_s, dec_s)
        if fast:
            K_2 = self.K_n_fast(ra_s, dec_s, 2)
            J_1 = self.J_n_fast(ra_s, dec_s, 1)
        else:
            K_2 = self.K_n(ra_s, dec_s, 2)
            J_1 = self.J_n(ra_s, dec_s, 1)
        K_2 = np.reshape(K_2, np.shape(ra_s))
        J_1 = np.reshape(J_1, np.shape(ra_s))
        xpp, ypp = self.prime_to_double_prime(xp, yp, self.A, self.B, self.C)
        return 2 * q * ypp ** 2 * K_2 + q * J_1

    def plot_q(self, ra_s, dec_s):
        theta = self.theta
        # print("theta is", theta)
        phi = self.phi
        f = self.f(theta, phi)
        A = self.A(theta, phi)
        B = self.B(theta, phi)
        C = self.C(theta, phi)
        qx = self.qx(f, A, B, C)
        qy = self.qy(f, A, B, C)
        q = self.q(qx, qy)
        return q

    def phi_x(self, xpp, ypp):
        '''(7) in keeton, arxiv: astro-ph/0102341v2'''
        q = self.q
        J_0 = self.J_n_inxy(xpp, ypp, 0)
        J_0 = np.reshape(J_0, np.shape(xpp))
        return q * J_0 * xpp

    def J_n_inxy(self, xpp, ypp, n):
        '''J_n but simplified to just take xpp ypp as arguments'''
        out = np.array([])
        xpp = np.ravel(xpp)
        ypp = np.ravel(ypp)
        for i in range(len(xpp)):
            temp = quad(self.J_n_integrand, 0, 1, args=(xpp[i], ypp[i], n))[0]
            out = np.append(out, temp)
        return out

    def phi_y(self, ypp, xpp):
        '''(8) in keeton, arxiv: astro-ph/0102341v2. Note that I used ypp as the first varible'''
        q = self.q
        J_1 = self.J_n(xpp, ypp, 1)
        J_1 = np.reshape(J_1, np.shape(xpp))
        return q * J_1 * ypp

    def phi_xy(self, ra_s, dec_s, fast=True):
        '''A15 in CK'''
        theta = self.theta
        # print("theta is", theta)
        q = self.q
        xp = self.cal_xp(ra_s, dec_s)
        yp = self.cal_yp(ra_s, dec_s)
        if fast:
            K_1 = self.K_n_fast(ra_s, dec_s, 1)
        else:
            K_1 = self.K_n(ra_s, dec_s, 1)
        K_1 = np.reshape(K_1, np.shape(xp))
        # print(K_1)
        xpp, ypp = self.prime_to_double_prime(xp, yp, self.A, self.B, self.C)
        # print(xpp)
        return 2 * q * ypp * xpp * K_1

    def sigma_hard_way(self, ra_s, dec_s):
        return 1 / 2 * (self.phi_xx(ra_s, dec_s) + self.phi_yy(ra_s, dec_s))

    def __DeltaSigmaComplex(self, x0, ra_s0, dec_s0):
        '''A10 in CK'''
        # no cutoff needed.
        ra_s = ra_s0
        dec_s = dec_s0
        phi_xx = self.phi_xx(ra_s, dec_s)
        phi_yy = self.phi_yy(ra_s, dec_s)
        phi_xy = self.phi_xy(ra_s, dec_s)
        true_phi_xx, true_phi_xy, true_phi_yy = self.convert_phi_doubleprime_prime(phi_xx, phi_xy, phi_yy, self.psi)
        gamma1 = 1 / 2 * (true_phi_xx - true_phi_yy)
        gamma2 = true_phi_xy
        # gamma1 = 1/2*(phi_xx - phi_yy)
        # gamma2 = phi_xy
        return gamma1 + 1j * gamma2

    def DeltaSigmaComplex(self, ra_s, dec_s):
        """Calculate Surface Density (Sigma) of halo.
                Takada & Jain(2003, MNRAS, 340, 580) Eq.27
                    ra_s:       ra of sources [arcsec].
                    dec_s:      dec of sources [arcsec].
                """

        # convenience: call with single number
        assert isinstance(ra_s, np.ndarray) == isinstance(dec_s, np.ndarray), \
            'ra_s and dec_s do not have same type'
        if not isinstance(ra_s, np.ndarray):
            ra_sA = np.array([ra_s], dtype='float')
            dec_sA = np.array([dec_s], dtype='float')
            return self.Sigma(ra_sA, dec_sA)[0]
        assert len(ra_s) == len(dec_s), \
            'input ra and dec have different length '
        x = self.DdRs(ra_s, dec_s)
        return self.__DeltaSigmaComplex(x, ra_s, dec_s)

    def DeltaSigma(self, ra_s, dec_s):
        """Calculate Surface Density (Sigma) of halo.
                        Takada & Jain(2003, MNRAS, 340, 580) Eq.27
                            ra_s:       ra of sources [arcsec].
                            dec_s:      dec of sources [arcsec].
                        """

        # convenience: call with single number
        assert isinstance(ra_s, np.ndarray) == isinstance(dec_s, np.ndarray), \
            'ra_s and dec_s do not have same type'
        if not isinstance(ra_s, np.ndarray):
            ra_sA = np.array([ra_s], dtype='float')
            dec_sA = np.array([dec_s], dtype='float')
            return self.Sigma(ra_sA, dec_sA)[0]
        assert len(ra_s) == len(dec_s), \
            'input ra and dec have different length '
        x = self.DdRs(ra_s, dec_s)
        return np.sqrt(np.absolute(self.__DeltaSigmaComplex(x, ra_s, dec_s)))

    def zeta_compare_23_27(self, ra_s, dec_s):
        xp = self.cal_xp(ra_s, dec_s)
        yp = self.cal_yp(ra_s, dec_s)
        # u = 1
        theta = self.theta
        phi = self.phi
        f = self.f
        A = self.A
        B = self.B
        C = self.C
        # qx = self.qx(f, A, B, C)
        # qy = self.qy(f, A, B, C)
        # q = self.q(qx, qy)
        # print((qx**2 + qy**2)/((4*(A+C)*f)/(B**2-4*A*C)))
        zeta_second = np.sqrt(1 / f * (A * xp ** 2 + B * xp * yp + C * yp ** 2))  ## eqn 27
        # zeta_thrid = np.sqrt(self.zeta2(u,xp,yp,qx,q)) ## eqn 32
        # zeta_fourth = np.sqrt(xp**2/(qx**2) + yp**2/(qy**2))
        # print(zeta_second/zeta_thrid)
        h = self.h_ols(theta, phi, xp, yp)
        g = self.g(theta, phi, xp, yp)
        zeta = self.zeta(h, g, f)
        return zeta / zeta_second

    def zeta_compare_23_32(self, ra_s, dec_s):
        xp = self.cal_xp(ra_s, dec_s)
        yp = self.cal_yp(ra_s, dec_s)
        u = 1
        theta = self.theta
        phi = self.phi
        f = self.f
        A = self.A
        B = self.B
        C = self.C
        qx = self.qx
        qy = self.qy
        q = self.q
        # print((qx**2 + qy**2)/((4*(A+C)*f)/(B**2-4*A*C)))
        zeta_second = np.sqrt(1 / f * (A * xp ** 2 + B * xp * yp + C * yp ** 2))  ## eqn 27
        zeta_thrid = np.sqrt(self.zeta2(u, xp, yp, qx, q))  ## eqn 32
        xpp, ypp = self.prime_to_double_prime(xp, yp, A, B, C)
        zeta_fourth = np.sqrt(xpp ** 2 / (qx ** 2) + ypp ** 2 / (qy ** 2))
        # print(zeta_second/zeta_thrid)
        h = self.h_ols(theta, phi, xp, yp)
        g = self.g(theta, phi, xp, yp)
        zeta = self.zeta(h, g, f)
        return zeta / zeta_fourth

    def convert_phi_doubleprime_prime(self, phi_xx, phi_xy, phi_yy, psi):
        '''Essentially multivariable calculus chain rule. Detailed can be found in section 3.6 of Gary's notes
        we will return the respective '''
        cospsi = np.cos(psi)
        sinpsi = np.sin(psi)
        true_phi_xx = phi_xx * cospsi ** 2 - 2 * sinpsi * cospsi * phi_xy + sinpsi ** 2 * phi_yy
        true_phi_xy = phi_xy * (cospsi ** 2 - sinpsi ** 2) + phi_xx * sinpsi * cospsi - phi_yy * sinpsi * cospsi
        true_phi_yy = phi_xx * sinpsi ** 2 + 2 * sinpsi * cospsi * phi_xy + cospsi ** 2 * phi_yy
        return true_phi_xx, true_phi_xy, true_phi_yy


class triaxialJS02_grid_mock(Cartesian):
    def __init__(self, parser):
        Cartesian.__init__(self, parser)
        self.ks2D = ksmap(self.ny, self.nx)
        return

    def add_halo(self, halo):
        lk = halo.lensKernel(self.zcgrid)
        sigma, ra, dec, nsamp = haloJS02SigmaAtom_mock_catalog(halo, self.scale, self.ny, self.nx, normalize=False)
        sigma = self.pixelize_data(ra, dec, np.ones(nsamp) / 10, sigma, method='FFT')[
            0]  # np.ones(nsamp) is just a random choice that gets the data pixelized.
        dsigma = self.ks2D.transform(sigma, inFou=False,
                                     outFou=False)  # in real space because no fourier space function available.
        shear = dsigma[None, :, :] * lk[:, None, None]
        kappa = sigma[None, :, :] * lk[:, None, None]
        return kappa, shear, sigma

    def add_halo_from_dsigma(self, halo, add_noise=False, shear_catalog_name='9347.fits'):
        lk = halo.lensKernel(self.zcgrid)
        dsigma, ra, dec, nsamp = haloJS02SigmaAtom_mock_catalog_dsigma(halo, self.scale, self.ny, self.nx,
                                                                       normalize=False)
        if add_noise:
            s19A = fits.open(shear_catalog_name)
            data = s19A[1].data
            s19A_table = Table(data)
            error1, error2 = make_mock(s19A_table)  # the realistic error from HSC shear catalog
            shear = dsigma[None, :] * lk[:, None]
            random_ints1, random_ints2 = np.random.randint(0, high=error1.size, size=shear.size), np.random.randint(0,
                                                                                                                    high=error1.size,
                                                                                                                    size=shear.size)
            dg1 = np.zeros(shear.size)
            dg2 = np.zeros(shear.size)
            for i in range(shear.size):  # compute errors
                dg1[i] = error1[random_ints1[i]]
                dg2[i] = error2[random_ints2[i]]
            dg1 = dg1.reshape(shear.shape)
            dg2 = dg2.reshape(shear.shape)
            shear = shear + dg1 + 1j * dg2  # added noise in this step
            shear_shape = (len(self.zcgrid))
            shearpixreal = np.zeros((len(self.zcgrid), self.ny, self.nx), dtype=np.float128)
            shearpixcomplex = np.zeros((len(self.zcgrid), self.ny, self.nx), dtype=np.float128)
            for i in range(len(self.zcgrid)):
                shearpixreal[i, :, :] = self.pixelize_data(ra, dec, np.ones(nsamp) / 10, shear[i].real, method='FFT')[0]
                shearpixcomplex[i, :, :] = \
                    self.pixelize_data(ra, dec, np.ones(nsamp) / 10, shear[i].imag, method='FFT')[0]
            shearpix = shearpixreal + 1j * shearpixcomplex
            return shearpix, np.std(np.abs(dg1 + 1j * dg2))
        else:
            dsigmapixreal = self.pixelize_data(ra, dec, np.ones(nsamp) / 10, dsigma.real, method='FFT')[0]
            dsigmapiximag = self.pixelize_data(ra, dec, np.ones(nsamp) / 10, dsigma.imag, method='FFT')[0]
            dsigmapix = dsigmapixreal + 1j * dsigmapiximag
            shearpix = dsigmapix[None, :, :] * lk[:, None, None]
            return shearpix

    def add_halo_noise(self, halo, shear_catalog_name='9347.fits'):
        lk = halo.lensKernel(self.zcgrid)
        sigma, ra, dec, nsamp = haloJS02SigmaAtom_mock_catalog(halo, self.scale, self.ny, self.nx, normalize=False)
        sigma = self.pixelize_data(ra, dec, np.ones(nsamp) / 10, sigma, method='FFT')[
            0]  # np.ones(nsamp) is just a random choice that gets the data pixelized.
        dsigma = self.ks2D.transform(sigma, inFou=False,
                                     outFou=False)  # in real space because no fourier space function available.
        shear = dsigma[None, :, :] * lk[:, None, None]
        kappa = sigma[None, :, :] * lk[:, None, None]
        s19A = fits.open(shear_catalog_name)
        data = s19A[1].data
        s19A_table = Table(data) # better way to read in the data
        error1, error2 = make_mock(s19A_table)  # the realistic error from HSC shear catalog
        random_ints1, random_ints2 = np.random.randint(0, high=error1.size, size=shear.size), np.random.randint(0,
                                                                                                                high=error1.size,
                                                                                                                size=shear.size)
        # essentially add a error in each pixel
        dg1 = np.zeros(shear.size)
        dg2 = np.zeros(shear.size)
        for i in range(shear.size):
            dg1[i] = error1[random_ints1[i]]
            dg2[i] = error2[random_ints2[i]]
        dg1 = dg1.reshape(shear.shape)
        dg2 = dg2.reshape(shear.shape)
        shear = shear + dg1 + 1j * dg2
        return kappa, shear, sigma, dg1, dg2, np.std(np.abs(dg1 + 1j * dg2))


def make_mock(dat):
    """Simulates shape noise and measurement error using HSC year1 shape catalog
    Args:
        dat (ndarray):  input year-1 shape catalog
    Returns:
        dg1 (ndarray):  noise on shear (first component)
        dg2 (ndarray):  noise on shear (second component)
    """

    def RotCatalog(e1, e2):
        """Rotates galaxy ellipticity
        Args:
            e1 (ndarray):  input ellipticity (first component)
            e2 (ndarray):  input ellipticity (second component)
        Returns:
            e1_rot (ndarray):  rotated ellipticity (first component)
            e2_rot (ndarray):  rotated ellipticity (second component)
        """
        phi = 2.0 * np.pi * np.random.rand(len(e1))
        cs = np.cos(phi)
        ss = np.sin(phi)
        e1_rot = e1 * cs + e2 * ss
        e2_rot = (-1.0) * e1 * ss + e2 * cs
        return e1_rot, e2_rot

    e1_ini = dat['ishape_hsm_regauss_e1']  # shape
    e2_ini = dat['ishape_hsm_regauss_e2']
    erms = dat['ishape_hsm_regauss_derived_rms_e']  # RMS of shape noise
    sigma_e2 = (dat['ishape_hsm_regauss_derived_shape_weight']) ** (-1) - erms ** 2
    esigma = np.sqrt(sigma_e2) * (2 * np.random.randint(0, 2, size=(
        sigma_e2.shape)) - 1)  # my dataset does not have this dat['ishape_hsm_regauss_derived_sigma_e'] # 1 sigma of measurment error
    eres = 1. - np.average(erms ** 2.)  # shear response (no shape weight)

    # rotate galaxy
    e1_rot, e2_rot = RotCatalog(e1_ini, e2_ini)

    # shape noise
    # equation (23) of https://arxiv.org/pdf/1901.09488.pdf
    f = np.sqrt(erms * erms / (erms * erms + esigma * esigma))
    e1_shape = e1_rot * f;
    e2_shape = e2_rot * f
    # measurment error
    e1_n = esigma * np.random.randn(len(e1_ini))
    e2_n = esigma * np.random.randn(len(e2_ini))
    dg1 = (e1_n + e1_shape) / 2. / eres
    dg2 = (e2_n + e2_shape) / 2. / eres
    return dg1, dg2


class prepare_numerical_frame(Cartesian):
    '''A Class that takes in parameters including redshifts, scale radius, (ellipticity may be in the future)to create
    FOURIER frames of SIGMA field. Important numbers are passed in from .ini files'''

    def __init__(self, parser, halo_mass, filename, alpha=1, fou=False):
        # halo_mass is the log mass of the halo.
        self.fou = fou
        Cartesian.__init__(self, parser)
        self.nframe = parser.getint('sparse', 'nframe')
        self.nzl = parser.getint('lens', 'nlp')
        if self.nzl <= 1:
            self.zlMin = 0.
            self.zlscale = 1.
        else:
            self.zlMin = parser.getfloat('lens', 'zlMin')
            self.zlscale = parser.getfloat('lens', 'zlscale')
        self.zlBin = zMeanBin(self.zlMin, self.zlscale, self.nzl)
        self.alpha = alpha
        self.ny = parser.getint('transPlane', 'ny')
        self.nx = parser.getint('transPlane', 'nx')
        self.rs_base = parser.getfloat('lens', 'rs_base')  # Mpc/h. I will keep using Mpc/h for length units.
        self.rsBin = np.zeros(self.nframe)
        self.sigmaAtom = np.zeros((self.nzl, self.nframe, self.ny, self.nx), dtype=float)
        self.fouaframesInter_real = np.zeros((self.nzl, self.nframe, self.ny, self.nx), dtype=float)
        self.fouaframesInter_complex = np.zeros((self.nzl, self.nframe, self.ny, self.nx), dtype=float)
        self.halo_mass = halo_mass  # an array of halo masses
        assert len(self.halo_mass) == self.nframe
        if parser.has_option('cosmology', 'omega_m'):
            omega_m = parser.getfloat('cosmology', 'omega_m')
        else:
            omega_m = Default_OmegaM
        self.cosmo = Cosmo(H0=Default_h0 * 100., Om0=omega_m)
        unit = parser.get('transPlane', 'unit')
        # Rescaling to degree
        if unit == 'degree':
            self.ratio = 1.
        elif unit == 'arcmin':
            self.ratio = 1. / 60.
        elif unit == 'arcsec':
            self.ratio = 1. / 60. / 60.
        self.scale = parser.getfloat('transPlane', 'scale') * self.ratio
        self.filename = filename
        return

    def __create_frames(self, long_truncation=False, OLS03=False):
        for izl in range(self.nzl):
            for im in reversed(range(len(self.halo_mass))):
                logm = self.halo_mass[im]
                M_200 = 10 ** logm
                if self.alpha == 1:
                    halo = triaxialJS02(mass=M_200, conc=4, redshift=self.zlBin[izl], ra=0., dec=0., a_over_c=1.0,
                                        a_over_b=1.0, tri_nfw=True,
                                        long_truncation=long_truncation,
                                        OLS03=OLS03)  # nfwWB00(mass=M_200, conc=4, redshift=self.zlBin[izl], ra=0., dec=0.)
                else:
                    halo = triaxialJS02(mass=M_200, conc=4, redshift=self.zlBin[izl], ra=0., dec=0., a_over_c=1.0,
                                        a_over_b=1.0, long_truncation=long_truncation, OLS03=OLS03)
                sigma, ra, dec, nsamp = haloJS02SigmaAtom_mock_catalog(halo, self.scale, self.ny, self.nx,
                                                                       normalize=False)
                sigma = self.pixelize_data(ra, dec, np.ones(nsamp) / 10., sigma, method='FFT')[0]
                self.sigmaAtom[izl, im] = sigma / M_200
        hdu1 = fits.PrimaryHDU(self.sigmaAtom)
        hdu1.writeto(self.filename)
        del hdu1
        return

    def __create_frames_fou(self):
        for izl in range(self.nzl):
            for im in reversed(range(len(self.halo_mass))):
                logm = self.halo_mass[im]
                M_200 = 10 ** logm
                if self.alpha == 1:
                    halo = nfwWB00(mass=M_200, conc=4, redshift=self.zlBin[izl], ra=0., dec=0.)
                else:
                    halo = triaxialJS02(mass=M_200, conc=4, redshift=self.zlBin[izl], ra=0., dec=0., a_over_c=1.0,
                                        a_over_b=1.0)
                sigma, ra, dec, nsamp = haloJS02SigmaAtom_mock_catalog(halo, self.scale, self.ny, self.nx,
                                                                       normalize=False)
                sigma = self.pixelize_data(ra, dec, np.ones(nsamp) / 10., sigma, method='FFT')[0]
                fousigma = np.fft.fft2(np.fft.fftshift(sigma))
                rpix = self.cosmo.angular_diameter_distance(self.zlBin[izl]).value / 180. * np.pi * self.scale
                znorm = 1. / rpix ** 2
                normTmp = fousigma[0, 0] / znorm
                fousigma = fousigma / normTmp
                self.rsBin[im] = halo.rs
                self.fouaframesInter_real[izl, im] = fousigma.real
                self.fouaframesInter_complex[izl, im] = fousigma.complex
        hdu1 = fits.PrimaryHDU(self.fouaframesInter_real)
        hdu1.writeto('fourier_real.fits')
        hdu2 = fits.PrimaryHDU(self.fouaframesInter_complex)
        hdu2.writeto('fourier_complex.fits')
        del hdu1
        del hdu2
        return

    def create_frames(self, long_truncation=False, OLS03=False):
        if self.fou:
            return self.__create_frames_fou()  # fourier space
        else:
            return self.__create_frames(long_truncation=long_truncation,
                                        OLS03=OLS03)  # configuration space. The preferred way.
