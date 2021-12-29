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
from configparser import ConfigParser

from splinv import hmod
from splinv.grid import Cartesian

def test_halolet():
    """ Test halolet with one frame
    """

    z_h     =   0.2425
    log_m   =   15.6
    M_200   =   10.**(log_m)
    conc    =   4.#6.02*(M_200/1.E13)**(-0.12)*(1.47/(1.+z_h))**(0.16)
    halo    =   hmod.nfwTJ03(mass=M_200,conc=conc,redshift=z_h,ra=0.,dec=0.)

    configName  =   'config_halolet.ini'
    parser      =   ConfigParser()
    parser.read(configName)

    # init sparse
    parser.set('lens','resolve_lim','0.1')  #pix
    parser.set('lens','rs_base','%s' %halo.rs)    #Mpc/h
    parser.set('sparse','nframe','1' )

    # init grid
    Grid=   Cartesian(parser)
    lensKer1=   Grid.lensing_kernel(deltaIn=False)
    L       =   Grid.nx*Grid.scale
    modelDict=  hmod.nfwShearlet2D(parser,lensKer1)

    xx=np.fft.fftshift(np.fft.fftfreq(Grid.nx,1./Grid.nx))
    yy=np.fft.fftshift(np.fft.fftfreq(Grid.ny,1./Grid.ny))
    XX,YY=np.meshgrid(xx,yy)


    np.testing.assert_almost_equal(modelDict.rs_frame[4],halo.rs_arcsec/Grid.scale/3600.,3)

    nsamp   =   2000000
    ra      =   np.random.rand(nsamp)*L-L/2.
    dec     =   np.random.rand(nsamp)*L-L/2.
    redshift=   np.ones(nsamp)*0.1
    sigma   =   halo.Sigma(ra*3600.,dec*3600.)
    data0   =   Grid.pixelize_data(ra,dec,redshift,sigma,\
                   method='FFT')[0]
    totest0 =   np.fft.fftshift(np.fft.ifft2(modelDict.fouaframesInter[4,0]).real)*M_200
    np.testing.assert_almost_equal(np.sum(data0)/np.sum(totest0),1,1)
    np.testing.assert_almost_equal(np.sum(totest0-data0)/np.sum(data0),0,2)


    dsigma  =   halo.DeltaSigmaComplex(ra*3600.,dec*3600.)
    g1Pix   =   Grid.pixelize_data(ra,dec,redshift,dsigma.real,\
                   method='FFT')[0]
    g2Pix   =   Grid.pixelize_data(ra,dec,redshift,dsigma.imag,\
                   method='FFT')[0]
    data1   =   np.abs(g1Pix+1j*g2Pix)
    totest1 =   np.abs(np.fft.fftshift(np.fft.ifft2(modelDict.fouaframes[4,0]))*M_200)

    np.testing.assert_almost_equal(np.sum(data1)/np.sum(totest1),1,1)
    e1  =   (np.sum(XX**2.*totest1)-np.sum(YY**2.*totest1))/np.sum(totest1)
    e2  =   np.sum(2.*XX*YY*totest1)/np.sum(totest1)
    np.testing.assert_almost_equal(e1,0,3)
    np.testing.assert_almost_equal(e2,0,3)
    res =   np.sum(np.abs(totest1-data1))/np.sum(data1)
    np.testing.assert_almost_equal(res,0,1)
    return

if __name__ == '__main__':
    test_halolet()
