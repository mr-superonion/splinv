#!/usr/bin/env python
import os
import fitsio
import numpy as np
from configparser import SafeConfigParser

isim        =   5  
lbd         =   5
glDir       =   'glResult%s-lambda%s/'%(isim,lbd)
dimen       =   '3D'
if not os.path.exists(glDir):
    os.mkdir(glDir)

simDir      =   'simulation%s/'%(isim)
assert os.path.exists(simDir),\
    "do not have input simulation files"
inSrc       =   os.path.join(simDir,'src.fits')

parser = SafeConfigParser()
configName  =   '/work/xiangchong.li/superonionGW/code/kappaMap_Private/config/s16a%sConfig.ini' %(dimen)
parser.read(configName)

ngList  =   [64]
for ngrid in ngList: #(pix)
    configName2 =   os.path.join(glDir,'s16aConfig%s_%d.ini' %(dimen,ngrid/2))
    size    =   64
    pix_scale=  size/ngrid
    parser.set('field','pixel_size','%s' % pix_scale)
    parser.set('field','npix','%s' % ngrid)
    parser.set('field','units','arcmin')
    parser.set('field','nlp','16')
    parser.set('survey','center_ra','%s' % (-pix_scale/2.))
    parser.set('survey','center_dec','%s' % (-pix_scale/2.))
    parser.set('survey','units','arcmin')
    parser.set('survey','size','64')
    parser.set('survey','z','z')
    parser.set('survey','zsig_min','0')
    parser.set('survey','zsig_max','0')
    parser.set('parameters','lambda','%s' % (lbd))
    parser.set('parameters','nrandom','100')
    parser.set('parameters','niter','500')
    parser.set('parameters','niter_debias','0')
    parser.set('parameters','nreweights','0')
    parser.set('parameters','nscales','7')
    with open(configName2, 'w') as configfile:
        parser.write(configfile)
    outFname    =   'kappaMap_GL_E_%s.fits' %(int(ngrid/2))
    outFname    =   os.path.join(glDir,outFname)
    binDir      =   '/work/xiangchong.li/superonionGW/packages/Glimpse/build'
    os.system('%s/glimpse %s %s %s' %(binDir,configName2,inSrc,outFname))
    """
    measMap     =   fitsio.read(outFname)
    truFname    =   os.path.join(simDir,'kMap_true%s.fits' %(int(ngrid/2)))
    truMap      =   fitsio.read(truFname)
    resFname    =   'kappaMap_GL_E_res_%s.fits' %(int(ngrid/2))
    resFname    =   os.path.join(glDir,resFname)
    resMap      =   measMap-truMap
    fitsio.write(resFname,resMap)
    """
