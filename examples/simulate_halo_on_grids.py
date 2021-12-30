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
import astropy.io.fits as pyfits
from splinv import hmod
from configparser import ConfigParser

def main():
    """ This is a simplified version of halo simulation described in
    Li et. al (2021, ApJ 916 67), which does not include photo-z
    uncertainties, pixelation effect due to sampling.
    """
    # configuration
    configName  =   'config_darkmapper.ini'
    parser      =   ConfigParser()
    parser.read(configName)

    # halo simulation
    CS02    =   hmod.nfwCS02_grid(parser)
    z_h     =   0.2425
    log_m   =   14.6
    M_200   =   10.**(log_m)
    conc    =   4.
    halo    =   hmod.nfwTJ03(mass=M_200,conc=conc,redshift=z_h,ra=0.,dec=0.)
    kappa,shear =   CS02.add_halo(halo)
    pyfits.writeto('k0_stamp.fits',kappa,overwrite=True)
    pyfits.writeto('g1_stamp.fits',shear.real,overwrite=True)
    pyfits.writeto('g2_stamp.fits',shear.imag,overwrite=True)
    return

if __name__ == '__main__':
    main()
