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
from splinv import detect
from splinv import hmod
from splinv import darkmapper
from splinv.grid import Cartesian
from configparser import ConfigParser

def main():
    # configuration
    configName  =   'config_darkmapper.ini'
    parser      =   ConfigParser()
    parser.read(configName)

    # Define halo
    z_h     =  0.2425           # halo redshift
    log_m   =  14.745           # halo mass (log10)
    M_200   =  10.**(log_m)     # halo mass
    conc    =  4.               # halo concentration
    halo    =  hmod.nfwTJ03(mass=M_200,conc=conc,redshift=z_h,ra=0.,dec=0.)

    # Reconstruction Init
    parser.set('sparse','mu','3e-4')            # learning rate
    parser.set('lens','resolve_lim','0.02')     # pix
    parser.set('lens','rs_base','%s' %halo.rs)  # Mpc/h
    parser.set('sparse','nframe','1' )          # number of NFW frame

    # Define the pixel grids
    # initialize the 3D Grids (z, dec, ra)
    # z is the index of redshift planes, dec and ra tell the position on the
    # transverse plane
    Grid    =   Cartesian(parser)
    # determine the lensing kernel for different redshfit planes of the grids
    lensKer =   Grid.lensing_kernel(deltaIn=False)

    # Simulation
    # assign each pixel in the grid with a shear distortion according to the
    # ra, dec and redshift of the pixel
    CS02    =   hmod.nfwCS02_grid(parser)
    data2   =   CS02.add_halo(halo)[1]
    # I use a simple error map here.
    # It doesnot matter since our data is noiseless
    gErr    =   np.ones(Grid.shape)*0.05

    dmapper =   darkmapper(parser,data2.real,data2.imag,gErr,lensKer)

    dmapper.lbd   =  8.     # lasso regularization parameter
    dmapper.lcd   =  0.     # ridge regularization parameter
    dmapper.nonNeg=  True   # using non-negative regularization
    dmapper.clean_outcomes()
    dmapper.fista_gradient_descent(3000)
    w   =   dmapper.adaptive_lasso_weight(gamma=2.)
    dmapper.fista_gradient_descent(3000,w=w)

    dmapper.mu=3e-3
    for _ in range(3):
        w   =   dmapper.adaptive_lasso_weight(gamma=2.)
        dmapper.fista_gradient_descent(3000,w=w)
    dmapper.reconstruct()
    c1  =   detect.local_maxima_3D(dmapper.deltaR)[0][0]
    np.testing.assert_equal(c1,np.array([4,Grid.ny//2,Grid.nx//2]))
    logm_est=   np.log10((dmapper.alphaR*dmapper._w)[4,0,Grid.ny//2,Grid.nx//2])+14.
    np.testing.assert_almost_equal(logm_est,log_m,1)
    return

if __name__ == '__main__':
    main()
