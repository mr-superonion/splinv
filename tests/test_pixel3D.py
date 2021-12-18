import numpy as np
from ddmap import halosim
from configparser import ConfigParser
from ddmap.pixel3D import cartesianGrid3D

# prepare the random sampling points
nsamp=800000
ra=np.random.rand(nsamp)*0.44-0.22
dec=np.random.rand(nsamp)*0.44-0.22
z=np.ones(nsamp)*0.1

# initialize pixel grids
parser      =   ConfigParser()
parser.read('test_pixel_trans.ini')
gridInfo    =   cartesianGrid3D(parser)


parser2     =   ConfigParser()
parser2.read('test_pixel_los.ini')
gridInfo2   =   cartesianGrid3D(parser2)

def test_pixel_transverse(log_m=15.,zh=0.3):
    '''Test cosistency between transverse plane pixelation and Fouarier-based nfw halo simulation
    '''

    # halo properties
    M_200=  10.**(log_m)
    conc =  6.02*(M_200/1.E13)**(-0.12)*(1.47/(1.+zh))**(0.16)
    # initialize halo
    halo =  halosim.nfw_lensTJ03(mass=M_200,conc=conc,redshift=zh,ra=0.,dec=0.)
    sigma=  halo.Sigma(ra*3600.,dec*3600.)
    dr  =   halo.angular_diameter_distance(zh).value*gridInfo.delta/180*np.pi

    pixSigma=gridInfo.pixelize_data(ra,dec,z,sigma)[0]/(M_200/dr**2.)
    # The (0,0) point is unstable
    pixSigma[gridInfo.ny//2,gridInfo.nx//2]=0.
    np.testing.assert_almost_equal(np.sum(pixSigma), 1, 2)

    rpix    =   halo.rs_arcsec/gridInfo.delta/3600.
    pixSigma2=  np.fft.fftshift(halosim.haloCS02SigmaAtom(rpix,ny=gridInfo.ny,nx=gridInfo.nx,\
            smooth_scale=-1,c=halo.c,fou=False,lnorm=1))
    vmax    =   pixSigma2[gridInfo.ny//2,gridInfo.nx//2]
    pixSigma2[gridInfo.ny//2,gridInfo.nx//2]=0.
    assert np.max(np.abs(pixSigma2-pixSigma))/vmax<5e-2

    return

def test_lensing_kernel():
    '''Test cosistency in lensing kernels
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


if __name__ == '__main__':
    test_WB00_Galsim(log_m=14.2,zh=0.11)
    test_WB00_Galsim(log_m=15.0,zh=0.25)
    test_line_of_sight()
