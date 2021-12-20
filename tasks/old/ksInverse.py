#!/usr/bin/env python
import os
import fitsio
import numpy as np
import kmapUtilities as kUtil

isim        =   '4-smooth08'
ksDir       =   'ksResult%s/' %(isim)
if not os.path.exists(ksDir):
    os.mkdir(ksDir)

simDir      =   'simulation%s/' %(isim.split('-')[0])
assert os.path.exists(simDir),\
    "do not have input simulation files"
sigmaSmooth =   0.8

ngList      =   [64,128,256]
for ngrid in ngList: #(pix)
    smoPix      =   sigmaSmooth/32.*ngrid
    kappaMap    =   fitsio.read(os.path.join(simDir,'kMap_true%s.fits' %(ngrid)))
    g1Map       =   fitsio.read(os.path.join(simDir,'g1Map_grid%s.fits'%(ngrid)))
    g2Map       =   fitsio.read(os.path.join(simDir,'g2Map_grid%s.fits'%(ngrid)))
    nMap        =   fitsio.read(os.path.join(simDir,'numMap_grid%s.fits'%(ngrid)))
    kE_KS,kB_KS =   kUtil.smooth_ksInverse(g1Map,g2Map,nMap,smoPix,0.2)
    fitsio.write(os.path.join(ksDir,'kappaMap_KS_E_%s.fits'%(ngrid)),kE_KS)
    fitsio.write(os.path.join(ksDir,'kappaMap_KS_B_%s.fits'%(ngrid)),kB_KS)
    fitsio.write(os.path.join(ksDir,'kappaMap_KS_E_res_%s.fits'%(ngrid)),kE_KS-kappaMap)

