#!/usr/bin/env python
import os
import fitsio
import numpy as np
from configparser import SafeConfigParser

isim        =   3  
lbd         =   4
glDir       =   'glResult%s-lambda4/'%(isim)
if not os.path.exists(glDir):
    os.mkdir(glDir)

simDir      =   'simulation%s/'%(isim)
assert os.path.exists(simDir),\
    "do not have input simulation files"

parser = SafeConfigParser()
parser.read('myConfig.ini')

ngList  =   [128,256,512]
for ngrid in ngList: #(pix)
    size    =   64
    pix_scale=  size/ngrid
    parser.set('field','pixel_size','%s' % pix_scale)
    parser.set('field','npix','%s' % ngrid)
    parser.set('survey','center_ra','%s' % (-pix_scale/2.))
    parser.set('survey','center_dec','%s' % (-pix_scale/2.))
    parser.set('parameters','lambda','%s' % (lbd))
    with open('myConfig.ini', 'w') as configfile:
        parser.write(configfile)
    outFname    =   'kappaMap_GL_E_%s.fits' %(int(ngrid/2))
    outFname    =   os.path.join(glDir,outFname)
    binDir      =   '/work/xiangchong.li/superonionGW/packages/Glimpse/build'
    os.system('%s/glimpse myConfig.ini %s %s' %(binDir,os.path.join(simDir,'src.fits'),outFname))
    measMap     =   fitsio.read(outFname)
    truFname    =   os.path.join(simDir,'kMap_true%s.fits' %(int(ngrid/2)))
    truMap      =   fitsio.read(truFname)
    resFname    =   'kappaMap_GL_E_res_%s.fits' %(int(ngrid/2))
    resFname    =   os.path.join(glDir,resFname)
    resMap      =   measMap-truMap
    fitsio.write(resFname,resMap)
