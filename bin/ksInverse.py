#!/usr/bin/env python
import os
import fitsio
import numpy as np
import kmapUtilities as kUtil

ksDir       =   'ksResult'
if not os.path.exists(ksDir):
    os.mkdir(ksDir)
simDir      =   './simulation'
assert os.path.exists(simDir),\
    "do not have input simulation files"

kappaMap    =   fitsio.read(os.path.join(simDir,'kMap_true.fits'))
g1Map       =   fitsio.read(os.path.join(simDir,'g1Map_true.fits'))
g2Map       =   fitsio.read(os.path.join(simDir,'g2Map_true.fits'))
mskMap      =   fitsio.read(os.path.join(simDir,'maskMap.fits'))
ny,nx       =   g1Map.shape
nxy         =   max(nx,ny)*2
g1Map       =   kUtil.zeroPad(g1Map,nxy)
g2Map       =   kUtil.zeroPad(g2Map,nxy)
mskMap      =   kUtil.zeroPad(mskMap,nxy)
kE_KS,kB_KS =   kUtil.ksInverse(g1Map,g2Map,mskMap)
kE_KS       =   kUtil.zeroPad_Inverse(kE_KS,nx,ny)
kB_KS       =   kUtil.zeroPad_Inverse(kB_KS,nx,ny)
fitsio.write(os.path.join(ksDir,'kappaMap_KS_E.fits'),kE_KS)
fitsio.write(os.path.join(ksDir,'kappaMap_KS_B.fits'),kB_KS)
fitsio.write(os.path.join(ksDir,'kappaMap_KS_E_res.fits'),kE_KS-kappaMap)


