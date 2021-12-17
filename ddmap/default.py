import numpy as np
from astropy import units

#important constant
C_LIGHT     =   2.99792458e8    # m/s
GNEWTON     =   6.67428e-11     # m^3/kg/s^2
KG_PER_SUN  =   1.98892e30      # kg/M_solar
M_PER_PARSEC=   3.08568025e16   # m/pc
Default_OmegaM= 0.315
Default_h0  =   1.              # set to 1
# rho~  [h^2]
# R~    [h^-1]
# V~    [h^-3]
# M~    [h^-1]


rho_unt =   units.solMass/units.Mpc**3.

def four_pi_G_over_c_squared():
    """
    4piG/c^2 = 1.5*H0^2/roh_0/c^2  [Mpc/M_solar]
    """
    fourpiGoverc2 = 4.0*np.pi*GNEWTON/(C_LIGHT**2)
    # in unit of pc/M_solar
    fourpiGoverc2 *= KG_PER_SUN/M_PER_PARSEC
    # in unit of Mpc/M_solar
    fourpiGoverc2 /= 1.e6
    return fourpiGoverc2
