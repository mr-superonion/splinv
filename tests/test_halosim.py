from ddmap import halosim
import numpy as np
from configparser import ConfigParser
from ddmap.pixel3D import cartesianGrid3D

try:
    import galsim
    has_galsim=True
except:
    has_galsim=False


def test_WB00_Galsim(log_m=15,zh=0.3):
    '''Test cosistency between WB00 and Galsim.nfw_halo
    '''
    if not has_galsim:
        print("do not have Galsim; therefore, we skip checking WB00 profile with galsim")
        return

    M_200   =   10**log_m
    conc    =   6.02*(M_200/1.E13)**(-0.12)*(1.47/(1.+zh))**(0.16)
    # create an WB00 halo
    halo=halosim.nfw_lensWB00(ra=0.,dec=0.,redshift=zh,mass=M_200,conc=conc)
    # create an galsim halo
    pos_cl  =   galsim.PositionD(0.,0.)
    haloGS  =   galsim.nfw_halo.NFWHalo(mass= M_200,
                conc=conc, redshift= zh,
                halo_pos=pos_cl ,omega_m= 0.3,
                omega_lam= 0.7)
    rlist=np.arange(0.01,1.5,0.05)
    nr=len(rlist)
    klist=np.empty(nr)
    slist=np.empty(nr)
    klistG=np.empty(nr)
    slistG=np.empty(nr)

    for i in range(nr):
        ratio=rlist[i]
        klist[i]=halo.lensKernel(2.)*halo.Sigma(2000.*ratio,1000.*ratio)
        slist[i]=halo.lensKernel(2.)*halo.DeltaSigmaComplex(2000.*ratio,1000.*ratio).real
        klistG[i]=haloGS.getConvergence(pos=(2000.*ratio,1000.*ratio),\
                           z_s=np.ones(1)*2.,units = "arcsec")
        slistG[i]=haloGS.getShear(pos=(2000.*ratio,1000.*ratio),\
                            z_s=np.ones(1)*2.,units = "arcsec",reduced=False)[0]

    assert np.max(np.abs(klist-klistG))/np.max(np.abs(klist))<1e-2
    assert np.max(np.abs(slist-slistG))/np.max(np.abs(slist))<1e-2
    return


def test_TJ03_Fourier(log_m=15.,zh=0.3):
    '''Test cosistency between TJ03 Fourier- and Real- based simulation
    '''

    # halo properties
    M_200=  10.**(log_m)
    conc =  6.02*(M_200/1.E13)**(-0.12)*(1.47/(1.+zh))**(0.16)

    # initialize halo
    halo =   halosim.nfw_lensTJ03(mass=M_200,conc=conc,redshift=zh,ra=0.,dec=0.)
    # initialize pixel grids
    configName  =   'test_halosim.ini'
    parser      =   ConfigParser()
    parser.read(configName)
    gridInfo    =   cartesianGrid3D(parser)
    yy,xx=np.meshgrid(gridInfo.ycgrid,gridInfo.xcgrid,indexing='ij')
    # get the surface density
    haloSigma2  =   halo.Sigma(xx.flatten()*3600.,yy.flatten()*3600.).reshape((gridInfo.ny,gridInfo.nx))
    # The (0,0) point is unstable
    haloSigma2[gridInfo.ny//2,gridInfo.nx//2]=0.
    # l2 normalization
    norm=   (np.sum(haloSigma2**2.))**0.5
    haloSigma2=haloSigma2/norm

    rpix    =   halo.rs_arcsec/gridInfo.delta/3600.
    haloSigma1= np.fft.fftshift(halosim.haloCS02SigmaAtom(rpix,ny=gridInfo.ny,nx=gridInfo.nx,c=halo.c,fou=False))
    # The (0,0) point is unstable
    haloSigma1[gridInfo.ny//2,gridInfo.nx//2]=0.
    # l2 normalization
    norm    =   (np.sum(haloSigma1**2.))**0.5
    haloSigma1= haloSigma1/norm
    vmax    =   haloSigma1.max()
    assert np.max(np.abs(haloSigma1-haloSigma2))<vmax/100.
    return

if __name__ == '__main__':
    test_WB00_Galsim(log_m=14.2,zh=0.11)
    test_WB00_Galsim(log_m=15.0,zh=0.30)
    test_WB00_Galsim(log_m=13.8,zh=0.08)
    # I am only testing for some well-sampled cases
    test_TJ03_Fourier(log_m=14.2,zh=0.11)
    test_TJ03_Fourier(log_m=15.0,zh=0.17)
    test_TJ03_Fourier(log_m=13.8,zh=0.08)
