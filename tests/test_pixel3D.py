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
import gc
import numpy as np
from splinv import hmod
from configparser import ConfigParser
from splinv.grid import Cartesian

# prepare the random sampling points
nsamp=800000
ra=np.random.rand(nsamp)*0.44-0.22
dec=np.random.rand(nsamp)*0.44-0.22
z=np.ones(nsamp)*0.1

# initialize pixel grids
parser      =   ConfigParser()
parser.read('config_pixel_trans.ini')
Grid    =   Cartesian(parser)

parserS     =   ConfigParser()
parser.read('config_pixel_trans_smooth.ini')
GridS   =   Cartesian(parser)

parser2     =   ConfigParser()
parser2.read('config_pixel_los.ini')
Grid2   =   Cartesian(parser2)

def test_pixel_transverse(log_m=15.,zh=0.3):
    '''Test cosistency between transverse plane pixelation and Fourier-based
    nfw halo simulation without smoothing
    '''

    # halo properties
    M_200=  10.**(log_m)
    conc =  6.02*(M_200/1.E13)**(-0.12)*(1.47/(1.+zh))**(0.16)
    # initialize halo
    halo =  hmod.nfwTJ03(mass=M_200,conc=conc,redshift=zh,ra=0.,dec=0.)
    sigma=  halo.Sigma(ra*3600.,dec*3600.)
    # angular scale of pixel size in Mpc
    dr  =   halo.DaLens*Grid.scale/180*np.pi

    pixSigma=Grid.pixelize_data(ra,dec,z,sigma)[0]/(M_200/dr**2.)
    # print(np.sum(pixSigma))
    # The (0,0) point is unstable due to aliasing
    pixSigma[Grid.ny//2,Grid.nx//2]=0.
    # print(np.sum(pixSigma))
    np.testing.assert_almost_equal(np.sum(pixSigma), 1, 2)

    rpix    =   halo.rs_arcsec/Grid.scale/3600.
    pixSigma2=  np.fft.fftshift(hmod.haloCS02SigmaAtom(rpix,ny=Grid.ny,nx=Grid.nx,\
            sigma_pix=-1,c=halo.c,fou=False,lnorm=1))
    vmax    =   pixSigma2[Grid.ny//2,Grid.nx//2]
    pixSigma2[Grid.ny//2,Grid.nx//2]=0.
    # print(np.max(np.abs(pixSigma2-pixSigma))/vmax)
    assert np.max(np.abs(pixSigma2-pixSigma))/vmax<5e-2

    return

def test_pixel_transverse_smooth(log_m=15.,zh=0.3):
    '''Test cosistency between transverse plane pixelation and Fourier-based
    nfw halo simulation with smoothing
    '''

    # halo properties
    M_200=  10.**(log_m)
    conc =  6.02*(M_200/1.E13)**(-0.12)*(1.47/(1.+zh))**(0.16)
    # initialize halo
    halo =  hmod.nfwTJ03(mass=M_200,conc=conc,redshift=zh,ra=0.,dec=0.)
    sigma=  halo.Sigma(ra*3600.,dec*3600.)
    # angular scale of pixel size in Mpc
    dr   =  halo.DaLens*GridS.scale/180*np.pi

    pixSigma=GridS.pixelize_data(ra,dec,z,sigma)[0]/(M_200/dr**2.)
    np.testing.assert_almost_equal(np.sum(pixSigma), 1, 2)

    rpix    =   halo.rs_arcsec/GridS.scale/3600.
    pixSigma2=  np.fft.fftshift(hmod.haloCS02SigmaAtom(rpix,ny=GridS.ny,nx=GridS.nx,\
            sigma_pix=GridS.sigma_pix,c=halo.c,fou=False,lnorm=1))
    vmax    =   pixSigma2[GridS.ny//2,GridS.nx//2]
    assert np.max(np.abs(pixSigma2-pixSigma))/vmax<5e-2

    return

def test_lensing_kernel():
    '''Test cosistency in lensing kernels in pixel3D
    '''
    lker1=Grid2.lensing_kernel(deltaIn=False)
    poz_grids=np.arange(0.01,2.1,0.01)
    ind=(Grid2.zcgrid-0.005)/0.01
    ind=ind.astype(np.int64)
    z_dens=np.zeros((Grid2.nz,len(poz_grids)))
    for i in range(Grid2.nz):
        z_dens[i,ind[i]]=1.
    lker2=Grid2.lensing_kernel(poz_grids=poz_grids,z_dens=z_dens,deltaIn=False)
    np.testing.assert_array_almost_equal(lker1,lker2,5)
    return

def test_lensing_kernel_halosim():
    '''Test cosistency in lensing kernels between pixel3D and halosim
    '''
    lker1=Grid2.lensing_kernel(deltaIn=False)
    lker3=[]
    for i in range(Grid2.nzl):
        halo=hmod.nfwWB00(ra=0.,dec=0.,\
            redshift=Grid2.zlcgrid[i],mass=1e14,conc=0.1)
        lker3.append(halo.lensKernel(Grid2.zcgrid))
        del halo
        gc.collect()
    lker3=np.stack(lker3).T*1e14
    np.testing.assert_array_almost_equal(lker3,lker1,5)
    return


if __name__ == '__main__':
    test_pixel_transverse(log_m=14.2,zh=0.11)
    test_pixel_transverse(log_m=15.0,zh=0.25)
    test_pixel_transverse_smooth(log_m=14.2,zh=0.11)
    test_pixel_transverse_smooth(log_m=15.0,zh=0.25)
    test_lensing_kernel()
    test_lensing_kernel_halosim()
