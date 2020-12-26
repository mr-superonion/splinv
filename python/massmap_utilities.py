import os
import warnings
import numpy as np
import healpy as hp

import astropy.io.fits as pyfits
from astropy.table import Table,join,vstack
from scipy.interpolate import griddata

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

