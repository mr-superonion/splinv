import numpy as np
from lintinv import detect
from lintinv import halosim
from lintinv import darkmapper
from lintinv.grid import Cartesian
from lintinv.halolet import ksmap
from configparser import ConfigParser

def test_sparse():
    """ Test sparse reconstruction of weak lensing dark map
    """
    # configuration
    configName  =   'config_sparse.ini'
    parser      =   ConfigParser()
    parser.read(configName)

    # halo simulation
    z_h     =  0.2425
    log_m   =  14.745
    M_200   =  10.**(log_m)
    conc    =  4.
    halo    =  halosim.nfw_lensTJ03(mass=M_200,conc=conc,redshift=z_h,ra=0.,dec=0.)

    # Reconstruction Init
    parser.set('sparse','mu','3e-4')
    parser.set('lens','resolve_lim','0.02')     #pix
    parser.set('lens','rs_base','%s' %halo.rs)  #Mpc/h
    parser.set('sparse','nframe','1' )

    # Pixelation
    gridInfo=   Cartesian(parser)
    Z,Y,X   =   np.meshgrid(gridInfo.zcgrid,gridInfo.ycgrid,gridInfo.xcgrid,indexing='ij')
    x,y,z   =   (X.flatten(),Y.flatten(),Z.flatten())
    lensKer1=   gridInfo.lensing_kernel(deltaIn=False)

    lk      =  halo.lensKernel(z)

    ks2D    =   ksmap(gridInfo.ny,gridInfo.nx)
    rpix    =   halo.rs_arcsec/gridInfo.scale/3600.
    sigma   =   halosim.haloCS02SigmaAtom(rpix,ny=gridInfo.ny,nx=gridInfo.nx,\
                sigma_pix=-1,c=halo.c,fou=True)
    snorm   =   sigma[0,0]
    dr      =   halo.DaLens*gridInfo.scale/180*np.pi
    snorm   =   M_200/dr**2./snorm
    sigma   =   sigma*snorm
    shear   =   np.fft.fftshift(ks2D.transform(sigma,inFou=True,outFou=False))
    lk2     =   lk.reshape(gridInfo.shape)
    data2   =   shear[None,:,:]*lk2
    gErr    =   np.ones(gridInfo.shape)*0.05

    dmapper =   darkmapper(parser,data2.real,data2.imag,gErr,lensKer1)

    dmapper.lbd=8.
    dmapper.lcd=0.
    dmapper.nonNeg=True
    dmapper.clean_outcomes()
    dmapper.fista_gradient_descent(3000)
    w   =   dmapper.adaptive_lasso_weight(gamma=2.)
    dmapper.fista_gradient_descent(3000,w=w)

    dmapper.mu=3e-3
    for i in range(3):
        w   =   dmapper.adaptive_lasso_weight(gamma=2.)
        dmapper.fista_gradient_descent(3000,w=w)
    dmapper.reconstruct()
    c1  =   detect.local_maxima_3D(dmapper.deltaR)[0][0]
    np.testing.assert_equal(c1,np.array([4,gridInfo.ny//2,gridInfo.nx//2]))
    logm_est=   np.log10((dmapper.alphaR*dmapper._w)[4,0,gridInfo.ny//2,gridInfo.nx//2])+14.
    np.testing.assert_almost_equal(logm_est,log_m,1)
    return

if __name__ == '__main__':
    test_sparse()
