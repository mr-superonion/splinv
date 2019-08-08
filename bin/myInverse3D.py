#!/usr/bin/env python
import os
import numpy as np
import astropy.io.fits as pyfits
from massmap_sparsity import massmap_sparsity_3D_2

isim        =   7
myDir       =   'myResult3D%s/' %(isim)
if not os.path.exists(myDir):
    os.mkdir(myDir)
simDir  =   './simulation%s/' %(isim)
sources =   pyfits.getdata(os.path.join(simDir,'src.fits'))
lbd     =   5
sparse3D=   massmap_sparsity_3D_2(sources,lbd=lbd,doDebug=True)
sparse3D.process()
outFname=   os.path.join(myDir,'kappaMap_3D_%s.fits'%(lbd))
pyfits.writeto(outFname, sparse3D.deltaR.real,overwrite=True)
