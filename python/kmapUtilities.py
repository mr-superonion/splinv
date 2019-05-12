import numpy as np

def zeroPad(oMap,nxy):
    nx      =   oMap.shape[1]
    ny      =   oMap.shape[0]
    xshift  =   (nxy-nx)//2
    yshift  =   (nxy-ny)//2
    pMap    =   np.zeros((nxy,nxy))
    pMap[yshift:yshift+ny,xshift:xshift+nx]=oMap
    return pMap

def zeroPad_Inverse(oMap,nxx,nyy):
    nx      =   oMap.shape[1]
    ny      =   oMap.shape[0]
    xshift  =   abs((nxx-nx)//2)
    yshift  =   abs((nyy-ny)//2)
    pMap    =   np.zeros((nyy,nxx))
    pMap    =   oMap[yshift:yshift+nyy,xshift:xshift+nxx]
    return pMap

def ksInverse(g1Map,g2Map,maskMap,isFour=False):
    gMap    =   g1Map+np.complex(0,1.)*g2Map
    if not isFour:
        gFouMap =   np.fft.fft2(gMap)
    else:
        gFouMap =   gMap.copy()
    e2phiF  =   e2phiFou(gFouMap.shape)
    kFouMap =   gFouMap/e2phiF/np.pi
    kMap    =   np.fft.ifft2(kFouMap)
    kEMap   =   kMap.real
    kBMap   =   kMap.imag
    kEMap   =   kEMap*maskMap
    kBMap   =   kBMap*maskMap
    return kEMap,kBMap

def e2phiFou(shape):
    ny1,nx1 =   shape
    e2phiF  =   np.zeros(shape,dtype=complex)
    for j in range(ny1):
        jy  =   (j+ny1//2)%ny1-ny1//2
        jy  =   jy/ny1
        for i in range(nx1):
            ix  =   (i+nx1//2)%nx1-nx1//2
            ix  =   ix/nx1
            if (i**2+j**2)>0:
                e2phiF[j,i]    =   np.complex((ix**2.-jy**2.),2.*ix*jy)/(ix**2.+jy**2.)
            else:
                e2phiF[j,i]    =   1.
    return e2phiF*np.pi
