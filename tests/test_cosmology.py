Default_h0=1.
omega_m=0.3

import astropy
import numpy as np
from astropy.cosmology import FlatLambdaCDM as Cosmo

try:
    import cosmology
    has_esheldon_cosmology=True
except:
    has_esheldon_cosmology=False

z       =   1.
cosmo   =   Cosmo(H0=Default_h0*100.,Om0=omega_m)
cosmo   =   Cosmo(H0=100,Om0=0.3)
uu=astropy.units.solMass/astropy.units.Mpc**3.
rhoc=cosmo.critical_density0.to_value(unit=uu)
rhoz=cosmo.critical_density(z).to_value(unit=uu)
rhomz=cosmo.critical_density(z).to_value(unit=uu)*cosmo.Om(z)
if has_esheldon_cosmology:
    cosmo2   =   cosmology.Cosmo(h=Default_h0,omega_m=omega_m)
    rhoc2=cosmo2.rho0()
    ezInv   =   cosmo2.Ez_inverse(z)
    # critical density (in unit of M_sun h^2 / Mpc^3)
    rhoz2   =   cosmo2.rho0()/ezInv**2
    rhomz2  =   cosmo2.rho_m(z)
    np.testing.assert_almost_equal((rhoc-rhoc2)/rhoc,0.,3)
    np.testing.assert_almost_equal((rhoz-rhoz2)/rhoz,0.,3)
