import numpy as np

def smooth_ksInverse(g1Map,g2Map,nMap,smoPix,mskThres=1.e-3):
    ny,nx       =   g1Map.shape
    nxy         =   max(nx,ny)*2
    g1Map       =   zeroPad(g1Map,nxy)
    g2Map       =   zeroPad(g2Map,nxy)
    nMap        =   zeroPad(nMap,nxy)
    mskMap      =   np.zeros((nxy,nxy)) 
    #do smoothing?
    if smoPix >= 1.:
        g1Map       =   smoothGaus(g1Map,smoPix)
        g2Map       =   smoothGaus(g2Map,smoPix)
        nMap        =   smoothGaus(nMap,smoPix)
    for j in range(nxy):
        for i in range(nxy):
            if nMap[j,i]>mskThres:
                g1Map[j,i]= g1Map[j,i]/nMap[j,i]
                g2Map[j,i]= g2Map[j,i]/nMap[j,i]
                mskMap[j,i]=1.
            else:
                g1Map[j,i]= 0. 
                g2Map[j,i]= 0.
                mskMap[j,i]=0.
    kE_KS,kB_KS =   ksInverse(g1Map,g2Map,mskMap)
    kE_KS       =   zeroPad_Inverse(kE_KS,nx,ny)
    kB_KS       =   zeroPad_Inverse(kB_KS,nx,ny)
    return kE_KS,kB_KS

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

def smoothGaus(inMap,sigma):
    fftMap  =   np.fft.fft2(inMap)
    nx  =   inMap.shape[1]
    ny  =   inMap.shape[0]
    assert nx==ny,\
        "different length for x and y axis"
    for j in range(ny):
        jy  =   (j+ny//2)%ny-ny//2
        jy  =   2.*np.pi*jy/ny
        for i in range(nx):
            ix  =   (i+nx//2)%nx-nx//2
            ix  =   2.*np.pi*ix/nx
            fftMap[j,i] =   fftMap[j,i]*np.exp(-(ix**2.+jy**2.)*sigma**2./2.)
    outMap  =   np.fft.ifft2(fftMap).real
    return outMap
    

def ksInverse(g1Map,g2Map,maskMap=None,isFour=False):
    gMap    =   g1Map+np.complex(0,1.)*g2Map
    if not isFour:
        gFouMap =   np.fft.fft2(gMap)
    else:
        gFouMap =   gMap.copy()
    e2phiF  =   e2phiFou(gFouMap.shape)
    kFouMap =   gFouMap/e2phiF*np.pi
    kMap    =   np.fft.ifft2(kFouMap)
    kEMap   =   kMap.real
    kBMap   =   kMap.imag
    if maskMap is not None:
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
