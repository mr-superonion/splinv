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
gridInfo    =   Cartesian(parser)

parserS     =   ConfigParser()
parser.read('config_pixel_trans_smooth.ini')
gridInfoS   =   Cartesian(parser)

parser2     =   ConfigParser()
parser2.read('config_pixel_los.ini')
gridInfo2   =   Cartesian(parser2)

def test_pixel_transverse(log_m=15.,zh=0.3):
    '''Test cosistency between transverse plane pixelation and Fourier-based
    nfw halo simulation without smoothing
    '''

    # halo properties
    M_200=  10.**(log_m)
    conc =  6.02*(M_200/1.E13)**(-0.12)*(1.47/(1.+zh))**(0.16)
    # initialize halo
    halo =  hmod.nfw_lensTJ03(mass=M_200,conc=conc,redshift=zh,ra=0.,dec=0.)
    sigma=  halo.Sigma(ra*3600.,dec*3600.)
    # angular scale of pixel size in Mpc
    dr  =   halo.DaLens*gridInfo.scale/180*np.pi

    pixSigma=gridInfo.pixelize_data(ra,dec,z,sigma)[0]/(M_200/dr**2.)
    # print(np.sum(pixSigma))
    # The (0,0) point is unstable due to aliasing
    pixSigma[gridInfo.ny//2,gridInfo.nx//2]=0.
    # print(np.sum(pixSigma))
    np.testing.assert_almost_equal(np.sum(pixSigma), 1, 2)

    rpix    =   halo.rs_arcsec/gridInfo.scale/3600.
    pixSigma2=  np.fft.fftshift(hmod.haloCS02SigmaAtom(rpix,ny=gridInfo.ny,nx=gridInfo.nx,\
            sigma_pix=-1,c=halo.c,fou=False,lnorm=1))
    vmax    =   pixSigma2[gridInfo.ny//2,gridInfo.nx//2]
    pixSigma2[gridInfo.ny//2,gridInfo.nx//2]=0.
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
    halo =  hmod.nfw_lensTJ03(mass=M_200,conc=conc,redshift=zh,ra=0.,dec=0.)
    sigma=  halo.Sigma(ra*3600.,dec*3600.)
    # angular scale of pixel size in Mpc
    dr   =  halo.DaLens*gridInfoS.scale/180*np.pi

    pixSigma=gridInfoS.pixelize_data(ra,dec,z,sigma)[0]/(M_200/dr**2.)
    np.testing.assert_almost_equal(np.sum(pixSigma), 1, 2)

    rpix    =   halo.rs_arcsec/gridInfoS.scale/3600.
    pixSigma2=  np.fft.fftshift(hmod.haloCS02SigmaAtom(rpix,ny=gridInfoS.ny,nx=gridInfoS.nx,\
            sigma_pix=gridInfoS.sigma_pix,c=halo.c,fou=False,lnorm=1))
    vmax    =   pixSigma2[gridInfoS.ny//2,gridInfoS.nx//2]
    assert np.max(np.abs(pixSigma2-pixSigma))/vmax<5e-2

    return

def test_lensing_kernel():
    '''Test cosistency in lensing kernels in pixel3D
    '''
    lker1=gridInfo2.lensing_kernel(deltaIn=False)
    poz_grids=np.arange(0.01,2.1,0.01)
    ind=(gridInfo2.zcgrid-0.005)/0.01
    ind=ind.astype(np.int64)
    z_dens=np.zeros((gridInfo2.nz,len(poz_grids)))
    for i in range(gridInfo2.nz):
        z_dens[i,ind[i]]=1.
    lker2=gridInfo2.lensing_kernel(poz_grids=poz_grids,z_dens=z_dens,deltaIn=False)
    np.testing.assert_array_almost_equal(lker1,lker2,5)
    return

def test_lensing_kernel_halosim():
    '''Test cosistency in lensing kernels between pixel3D and halosim
    '''
    lker1=gridInfo2.lensing_kernel(deltaIn=False)
    lker3=[]
    for i in range(gridInfo2.nzl):
        halo=hmod.nfw_lensWB00(ra=0.,dec=0.,\
            redshift=gridInfo2.zlcgrid[i],mass=1e14,conc=0.1)
        lker3.append(halo.lensKernel(gridInfo2.zcgrid))
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
