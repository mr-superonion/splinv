import os
import json
import cosmology
import numpy as np
from configparser import ConfigParser

C_LIGHT=2.99792458e8        # m/s
GNEWTON=6.67428e-11         # m^3/kg/s^2
KG_PER_SUN=1.98892e30       # kg/M_solar
M_PER_PARSEC=3.08568025e16  # m/pc

def four_pi_G_over_c_squared():
    # = 1.5*H0^2/roh_0/c^2
    # We want it return 4piG/c^2 in unit of Mpc/M_solar
    # in unit of m/kg
    fourpiGoverc2 = 4.0*np.pi*GNEWTON/(C_LIGHT**2)
    # in unit of pc/M_solar
    fourpiGoverc2 *= KG_PER_SUN/M_PER_PARSEC
    # in unit of Mpc/M_solar
    fourpiGoverc2 /= 1.e6
    return fourpiGoverc2

class cartesianGrid3D():
    # pixel3D.cartesianGrid3D
    def __init__(self,parser):
        # Transverse celestial plane
        unit=parser.get('transPlane','unit')
        if unit=='degree':
            ratio=1.
        elif unit=='arcmin':
            ratio=1./60.
        elif unit=='arcsec':
            ratio=1./60./60.

        ## ra
        delta=parser.getfloat('transPlane','scale')*ratio
        xmin=parser.getfloat('transPlane','xmin')*ratio
        nx=parser.getint('transPlane','nx')
        assert nx>=1
        xmax=xmin+delta*(nx+0.1)
        xbound=np.arange(xmin,xmax,delta)
        xcgrid=(xbound[:-1]+xbound[1:])/2.
        assert len(xcgrid)==nx
        self.xbound=xbound
        self.xcgrid=xcgrid
        ## dec
        ymin=parser.getfloat('transPlane','ymin')*ratio
        ny=parser.getint('transPlane','ny')
        assert ny>=1
        ymax=ymin+delta*(ny+0.1)
        ybound=np.arange(ymin,ymax,delta)
        ycgrid=(ybound[:-1]+ybound[1:])/2.
        assert len(ycgrid)==ny
        self.ybound=ybound
        self.ycgrid=ycgrid
        self.delta=delta
        ## Gaussian smoothing
        self.sigma=parser.getfloat('transPlane','smooth_scale')*ratio

        # line-of-signt direction
        # background plane
        if parser.has_option('sourceZ','zbound'):
            zbound=np.array(json.loads(parser.get('sourceZ','zbound')))
            nz=len(zbound)-1
        else:
            zmin=parser.getfloat('sourceZ','zmin')
            deltaz=parser.getfloat('sourceZ','zscale')
            nz=parser.getint('sourceZ','nz')
            assert nz>=1
            zmax=zmin+deltaz*(nz+0.1)
            zbound=np.arange(zmin,zmax,deltaz)

        zcgrid=(zbound[:-1]+zbound[1:])/2.
        self.zbound=zbound
        self.zcgrid=zcgrid
        self.nz=nz
        self.shape=(nz,ny,nx)

        # foreground plane
        if parser.has_option('lensZ','zlbound'):
            zlbound=np.array(json.loads(parser.get('sourceZ','zlbound')))
            nzl=len(zlbound)-1
        else:
            zlmin=parser.getfloat('lensZ','zlmin')
            deltazl=parser.getfloat('lensZ','zlscale')
            nzl=parser.getint('lensZ','nlp')
            assert nzl>=1
            zlmax=zlmin+deltazl*(nzl+0.1)
            zlbound=np.arange(zlmin,zlmax,deltazl)
        zlcgrid=(zlbound[:-1]+zlbound[1:])/2.
        self.zlbound=zlbound
        self.zlcgrid=zlcgrid
        self.nzl=nzl

        self.cosmo=cosmology.Cosmo(h=1,omega_m=0.3)
        self.lensKernel=None
        self.pozPdfAve=None

    def pixelize_data(self,x,y,z,v,ws=None):
        if ws is None:
            ws=np.ones(len(x))/0.25**2.
        xbin=np.int_((x-self.xbound[0])/self.delta)
        ybin=np.int_((y-self.ybound[0])/self.delta)
        if z is None:
            # This is for 2D pixelaztion
            assert self.shape[0]==1
        dataOut=np.zeros(self.shape,v.dtype)
        varOut=np.zeros(self.shape,dtype=float)
        # sample to 3 times of sigma
        if self.sigma>0:
            rsig=int(self.sigma/self.delta*3+1)
        else:
            return
        for iz in range(self.shape[0]):
            if z is not None:
                mskz=(z>=self.zbound[iz])&(z<self.zbound[iz+1])
            else:
                mskz=np.ones(len(x),dtype=bool)
            for iy in range(self.shape[1]):
                yc=self.ycgrid[iy]
                msky=mskz&(ybin>=iy-rsig)&(ybin<=iy+rsig)
                for ix in range(self.shape[2]):
                    xc=self.xcgrid[ix]
                    mskx=msky&(xbin>=ix-rsig)&(xbin<=ix+rsig)
                    if not np.sum(mskx)>=1:
                        continue
                    rl2=(x[mskx]-xc)**2.+(y[mskx]-yc)**2.
                    # convolve with Gaussian kernel
                    wg=np.exp(-rl2/self.sigma**2./2.)
                    wgsum=np.sum(wg)
                    wl=wg*ws[mskx]
                    if wgsum>2.:
                        dataOut[iz,iy,ix]=np.sum(wl*v[mskx])/np.sum(wl)
                        #varOut[iz,iy,ix]=2.*np.sum(wg**2.*ws[mskx])/(np.sum(wl))**2.
                        # 2 is to account for 2 components of shear (g1 and g2)
        return dataOut

    def lensing_kernel(self,poz_grids=None,poz_data=None,poz_best=None,poz_ave=None):
        # Mapping from an average delta in a lens redshfit bin
        # to an average kappa in a source redshift
        assert (poz_data is not None)==(poz_best is not None), \
            'Please provide both photo-z bins and photo-z data'
        assert (self.nzl==1)==(self.nz==1), \
            'number of lens plane and source plane'
        if self.nzl<=1:
            return np.ones((self.nz,self.nzl))
        lensKernel =   np.zeros((self.nz,self.nzl))
        if poz_grids is None and poz_ave is None:
            # Do not use poz
            for i,zl in enumerate(self.zlcgrid):
                kl =   np.zeros(self.nz)
                mask=  (zl<self.zcgrid)
                kl[mask] =   self.cosmo.Da(zl,self.zcgrid[mask])*self.cosmo.Da(0.,zl)\
                        /self.cosmo.Da(0.,self.zcgrid[mask])
                kl*=four_pi_G_over_c_squared()
                # Sigma_M_zl_bin
                # Surface masss density in lens bin
                rhoM_ave=self.cosmo.rho_m(zl)
                DaBin=self.cosmo.Da(self.zlbound[i],self.zlbound[i+1])
                lensKernel[:,i]=kl*rhoM_ave*DaBin
        else:
            # Use poz
            lensK =   np.zeros((len(poz_grids),self.nzl))
            for i,zl in enumerate(self.zlcgrid):
                kl=np.zeros(len(poz_grids))
                mask= (zl<poz_grids)
                #Dsl*Dl/Ds
                kl[mask] =   self.cosmo.Da(zl,poz_grids[mask])*self.cosmo.Da(0.,zl)/self.cosmo.Da(0.,poz_grids[mask])
                kl*=four_pi_G_over_c_squared()
                # Sigma_M_zl_bin
                rhoM_ave=self.cosmo.rho_m(zl)
                DaBin=self.cosmo.Da(self.zlbound[i],self.zlbound[i+1])
                lensK[:,i]=kl*rhoM_ave*DaBin

            # Prepare the poz average
            if poz_ave is None:
                assert len(poz_data)==len(poz_best)
                assert len(poz_data[0])==len(poz_grids)

                # determine the average photo-z uncertainty
                poz_ave=np.zeros((self.nz,len(poz_grids)))
                for iz in range(self.nz):
                    tmp_msk=(poz_best>=self.zbound[iz])&(poz_best<self.zbound[iz+1])
                    poz_ave[iz,:]=np.average(poz_data[tmp_msk],axis=0)
            self.poz_ave=poz_ave

            lensKernel=poz_ave.dot(lensK)
            self.lensKernel=lensKernel
        return lensKernel
