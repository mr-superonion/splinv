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
import warnings
import numpy as np
import healpy as hp

import astropy.io.fits as pyfits
from astropy.table import Table,join,vstack
from scipy.interpolate import griddata
import cosmology

# Field keys used in the noise variance file for simulations
field_keys = {'W02'     : 'XMM',
              'W03'     : 'GAMA09H',
              'W04-12'  : 'WIDE12H',
              'W04-15'  : 'GAMA15H',
              'W05'     : 'VVDS',
              'W06'     : 'HECTOMAP'}

field_names = {'XMM'    : 'W02',
              'GAMA09H' : 'W03',
              'WIDE12H' : 'W04_WIDE12H',
              'GAMA15H' : 'W04_GAMA15H',
              'VVDS'    : 'W05',
              'HECTOMAP': 'W06'}


def rotCatalog(e1, e2, phi=None):
    cs      =   np.cos(phi)
    ss      =   np.sin(phi)
    e1_rot  =   e1 * cs + e2 * ss
    e2_rot  =   (-1.0) * e1 * ss + e2 * cs
    return e1_rot, e2_rot

def nfwMrs2delta(z,M,rs,omega_m=0.3):
    """
    @param mass         Mass defined using a spherical overdensity of 200 times the critical density
                        of the universe, in units of M_solar/h.
    @param conc         Concentration parameter, i.e., ratio of virial radius to NFW scale radius.
    @param redshift     Redshift of the halo.
    @param omega_m      Omega_matter to pass to Cosmology constructor. [default: 0.3]
                        omega_lam is set to 1-omega_matter.
    """
    # Redshift and Geometry
    ## ra dec
    cosmo   =   cosmology.Cosmo(h=1,omega_m=omega_m)
    a       =   1./(1.+z)
    DaLens  =   cosmo.Da(0.,z) # angular distance in Mpc/h
    # E(z)^{-1}
    ezInv   =   cosmo.Ez_inverse(z)
    # critical density
    # in unit of M_solar / Mpc^3
    rho_cZ  =   cosmo.rho0()/ezInv**2
    # First, we get the virial radius, which is defined for some spherical
    # overdensity as 3 M / [4 pi (r_vir)^3] = overdensity Here we have
    # overdensity = 200 * rhocrit, to determine r_vir (angular distance).
    # The factor of 1.63e-5 comes from the following set of prefactors: (3
    # / (4 pi * 200 * rhocrit))^(1/3), where rhocrit = 2.8e11 h^2 M_solar /
    # Mpc^3.
    # (H0=100,DH=C_LIGHT/1e3/H0,rho_crit0=1.5/four_pi_G_over_c_squared()/(DH)**2.)
    rvir    =   1.63e-5*(M*ezInv**2)**(1./3.) # in Mpc/h
    c       =   rvir/rs
    # \delta_c in equation (2)
    A       =   1./(np.log(1+c)-(c)/(1+c))
    delta_nfw   =   200./3*c**3*A
    # convert to angular radius in unit of arcsec
    scale       =   rs / DaLens
    arcsec2rad  =   np.pi/180./3600
    rs_arcsec   =   scale/arcsec2rad

    # Second, derive the charateristic matter density
    # within virial radius at redshift z
    rho_s   =   rho_cZ*delta_nfw
    return rho_s
