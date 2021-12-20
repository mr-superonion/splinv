#!/usr/bin/env python
import os
import numpy as np
import astropy.io.fits as pyfits
from massmap_sparsity import massmap_sparsity_2D

isim    =   6
npix    =   128
g1Map   =   pyfits.getdata('simulation%s/g1Map_grid%s.fits' %(isim,npix))
g2Map   =   pyfits.getdata('simulation%s/g2Map_grid%s.fits' %(isim,npix))
nMap    =   pyfits.getdata('simulation%s/numMap_grid%s.fits'%(isim,npix))
ny,nx   =   nMap.shape
for j in range(ny):
    for i in range(nx):
        if nMap[j,i]>0.1:
            g1Map[j,i]= g1Map[j,i]/nMap[j,i]
            g2Map[j,i]= g2Map[j,i]/nMap[j,i]
        else:
            g1Map[j,i]= 0. 
            g2Map[j,i]= 0.
shearMap= g1Map+np.complex128(1j)*g2Map

myDir       =   'myResult2D%s/' %(isim)
if not os.path.exists(myDir):
    os.mkdir(myDir)
for lbd  in range(2,6):
    outFname=   os.path.join(myDir,'kappaMap_2D_%s.fits'%(lbd))
    sparse2D=   massmap_sparsity_2D(shearMap,nMap,lbd=lbd)
    sparse2D.process()
    pyfits.writeto(outFname, sparse2D.kappaR.real,overwrite=True)
