import numpy as np
from configparser import ConfigParser

class cartesianGrid3D():
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
        xmax=xmin+delta*(nx+0.1)
        xbound=np.arange(xmin,xmax,delta)-0.5*delta
        xcgrid=(xbound[:-1]+xbound[1:])/2.
        assert len(xcgrid)==nx
        self.xbound=xbound
        self.xcgrid=xcgrid
        ## dec
        ymin=parser.getfloat('transPlane','ymin')*ratio
        ny=parser.getint('transPlane','ny')
        ymax=ymin+delta*(ny+0.1)
        ybound=np.arange(ymin,ymax,delta)-0.5*delta
        ycgrid=(ybound[:-1]+ybound[1:])/2.
        assert len(ycgrid)==ny
        self.ybound=ybound
        self.ycgrid=ycgrid
        self.delta=delta
        ## Gaussian smoothing
        self.sigma=parser.getfloat('transPlane','smooth_scale')*ratio

        # line-of-signt direction
        zmin=parser.getfloat('sourceZ','zmin')
        deltaz=parser.getfloat('sourceZ','zscale')
        nz=parser.getint('sourceZ','nz')
        zmax=zmin+deltaz*(nz+0.1)
        zbound=np.arange(zmin,zmax,deltaz)-0.5*deltaz
        zcgrid=(zbound[:-1]+zbound[1:])/2.
        assert len(zcgrid)==nz
        self.zbound=zbound
        self.zcgrid=zcgrid
        self.deltaz=deltaz
        self.shape=(nz,ny,nx)

    def pixelize_data(self,x,y,z,v):
        xbin=np.int_((x-self.xbound[0])/self.delta)
        ybin=np.int_((y-self.ybound[0])/self.delta)
        if z is None:
            assert self.shape[0]==1
            zbin=np.zeros(len(x),dtype=int)
        else:
            zbin=np.int_((z-self.zbound[0])/self.deltaz)
        dataOut=np.zeros(self.shape)
        rsig=int(self.sigma/self.delta*3+1)
        for iz in range(self.shape[0]):
            zc=self.zcgrid[iz]
            mskz=(zbin==iz)
            for iy in range(self.shape[1]):
                yc=self.ycgrid[iy]
                msky=mskz&(ybin>=iy-rsig)&(ybin<=iy+rsig)
                for ix in range(self.shape[2]):
                    xc=self.xcgrid[ix]
                    mskx=msky&(xbin>=ix-rsig)&(xbin<=ix+rsig)
                    rl2=(x[mskx]-xc)**2.+(y[mskx]-yc)**2.
                    wl=1./np.sqrt(2.*np.pi)/self.sigma*np.exp(-rl2/self.sigma**2./2.)
                    wsum=np.sum(wl)
                    if wsum>0.1:
                        dataOut[iz,iy,ix]=np.sum(wl*v[mskx])/wsum
        return dataOut
    def lensing_kernel(self,poz_bins,poz_data):
        if poz_bins is None and poz_data is None:
            pass
        else:
            pass
        return
