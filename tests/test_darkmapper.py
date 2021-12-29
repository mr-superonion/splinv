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
from splinv.hmod import ksmap
from configparser import ConfigParser

def test_darkmapper():
    """ Test sparse reconstruction of weak lensing dark map
    """
    # configuration
    configName  =   'config_darkmapper.ini'
    parser      =   ConfigParser()
    parser.read(configName)

    # halo simulation
    z_h     =  0.2425
    log_m   =  14.745
    M_200   =  10.**(log_m)
    conc    =  4.
    halo    =  hmod.nfwTJ03(mass=M_200,conc=conc,redshift=z_h,ra=0.,dec=0.)

    # Reconstruction Init
    parser.set('sparse','mu','3e-4')
    parser.set('lens','resolve_lim','0.02')     #pix
    parser.set('lens','rs_base','%s' %halo.rs)  #Mpc/h
    parser.set('sparse','nframe','1' )

    # Pixelation
    Grid    =   Cartesian(parser)
    lensKer1=   Grid.lensing_kernel(deltaIn=False)


    CS02    =   hmod.nfwCS02_grid(parser)
    data2   =   CS02.add_halo(halo)[1]
    gErr    =   np.ones(Grid.shape)*0.05

    dmapper =   darkmapper(parser,data2.real,data2.imag,gErr,lensKer1)

    dmapper.lbd=8.
    dmapper.lcd=0.
    dmapper.nonNeg=True
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
    test_darkmapper()
