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

