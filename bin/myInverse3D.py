#!/usr/bin/env python
import os
import numpy as np
import astropy.io.fits as pyfits
from massmap_sparsity import massmap_sparsity_2D

isim    =   6
myDir       =   'myResult3D%s/' %(isim)
if not os.path.exists(myDir):
    os.mkdir(myDir)
    simDir  =   './simulation%s/' %(isim)
    sources =   pyfits.getdata(os.path.join(simDir,'src.fits'))
    size    =   32 #(arcmin)
    ngrid   =   64
    pix_scale   =   size/ngrid#(arcmin/pix)
    ngrid2  =   ngrid*2
    nlp     =   4
    z_scale =   0.5
    g1Map   =   np.zeros((nlp,ngrid2,ngrid2))
    g2Map   =   np.zeros((nlp,ngrid2,ngrid2))
    numMap  =   np.zeros((nlp,ngrid2,ngrid2),dtype=np.int)  
    xMin    =   -size
    yMin    =   -size
    zMin    =   0.02

    for ss in sources:
        ix  =   int((ss['ra']-xMin)//pix_scale)
        iy  =   int((ss['dec']-yMin)//pix_scale)
        iz  =   int((ss['z']-zMin)//z_scale)
        if iz>=0 and iz<4:
            g1Map[iz,iy,ix]    =   g1Map[iz,iy,ix]+ss['g1']
            g2Map[iz,iy,ix]    =   g2Map[iz,iy,ix]+ss['g2']
            numMap[iz,iy,ix]   =   numMap[iz,iy,ix]+1.
    mask    =   (numMap>0.1)
    g1Map[mask]=g1Map[mask]/numMap[mask]
    g2Map[mask]=g2Map[mask]/numMap[mask]
    pyfits.writeto(os.path.join(myDir,'g13D.fits'),g1Map)
    pyfits.writeto(os.path.join(myDir,'g23D.fits'),g2Map)
    pyfits.writeto(os.path.join(myDir,'num3D.fits'),numMap)
else:
    g1Map   =   pyfits.getdata(os.path.join(myDir,'g13D.fits'))
    g2Map   =   pyfits.getdata(os.path.join(myDir,'g23D.fits'))
    numMap  =   pyfits.getdata(os.path.join(myDir,'num3D.fits'))
shearMap= g1Map+np.complex128(1j)*g2Map

"""
lbd =   4
outFname=   os.path.join(myDir,'kappaMap_2D_%s.fits'%(lbd))
sparse3D=   massmap_sparsity_3D(shearMap,numMap,lbd=lbd)
sparse3D.process()
pyfits.writeto(outFname, sparse2D.kappaR.real,overwrite=True)
"""
