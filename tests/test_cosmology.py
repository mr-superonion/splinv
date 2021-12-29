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
import astropy.units as astunits
from astropy.cosmology import FlatLambdaCDM as Cosmo

try:
    import cosmology
    has_esheldon_cosmology=True
except:
    has_esheldon_cosmology=False

def test_cosmo(h0=1.0,omega_m=0.3):
    z       =   1.
    cosmo   =   Cosmo(H0=h0*100.,Om0=omega_m)
    uu      =   astunits.solMass/(astunits.Mpc)**3.
    rhoc    =   cosmo.critical_density0.to_value(unit=uu)
    rhoz    =   cosmo.critical_density(z).to_value(unit=uu)
    rhomz   =   cosmo.critical_density(z).to_value(unit=uu)*cosmo.Om(z)
    if has_esheldon_cosmology:
        cosmo2   =   cosmology.Cosmo(h=h0,omega_m=omega_m)
        rhoc2=cosmo2.rho0()
        ezInv   =   cosmo2.Ez_inverse(z)
        # critical density (in unit of M_sun h^2 / Mpc^3)
        rhoz2   =   cosmo2.rho0()/ezInv**2
        rhomz2  =   cosmo2.rho_m(z)
        np.testing.assert_almost_equal((rhoc-rhoc2)/rhoc,0.,3)
        np.testing.assert_almost_equal((rhoz-rhoz2)/rhoz,0.,3)
        np.testing.assert_almost_equal((rhomz-rhomz2)/rhoz,0.,3)
    return

if __name__ == '__main__':
    test_cosmo(h0=1.0,omega_m=0.3)
    test_cosmo(h0=0.7,omega_m=0.315)
