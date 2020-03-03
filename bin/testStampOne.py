#!/usr/bin/env python
import os
import numpy as np
import astropy.io.fits as pyfits
from configparser import ConfigParser
from sparseBase import massmapSparsityTask

def pixelize(simSrc,parser):
    fieldName   =   parser.get('file','fieldN')
    pixDir      =   parser.get('file','pixDir')
    g1Fname     =   os.path.join(pixDir,'g1Map_%s.fits' %fieldName)
    g2Fname     =   os.path.join(pixDir,'g2Map_%s.fits' %fieldName)
    nFname      =   os.path.join(pixDir,'nMap_%s.fits'  %fieldName)

    if os.path.exists(nFname):
        print('Alreadying have pixelized shear field')
        return

    if parser.has_option('transPlane','raname'):
        raname  =   parser.get('transPlane','raname')
    else:
        raname  =   'ra'
    if parser.has_option('transPlane','decname'):
        decname=   parser.get('transPlane','decname')
    else:
        decname =   'dec'
    if parser.has_option('sourceZ','zname'):
        zname   =   parser.get('sourceZ','zname')
    else:
        zname   =   'z'

    g1name  =   'g1'
    g2name  =   'g2'

    xMin    =   parser.getfloat('transPlane','xMin')
    yMin    =   parser.getfloat('transPlane','yMin')
    scale   =   parser.getfloat('transPlane','scale')
    ny      =   parser.getint('transPlane'  ,'ny')
    nx      =   parser.getint('transPlane'  ,'nx')
    zMin    =   parser.getfloat('sourceZ','zMin')
    zscale  =   parser.getfloat('sourceZ','zscale')
    nz      =   parser.getint('sourceZ','nz')
    shapeS  =   (nz,ny,nx)
    nMap    =   np.zeros(shapeS)
    g1Map   =   np.zeros(shapeS)
    g2Map   =   np.zeros(shapeS)
    for ss in simSrc:
        ix  =   int((ss[raname]-xMin)//scale)
        iy  =   int((ss[decname]-yMin)//scale)
        iz  =   int((ss[zname]-zMin)//zscale)
        if iz>=0 and iz<nz:
            g1Map[iz,iy,ix]=   g1Map[iz,iy,ix]  +   ss[g1name]
            g2Map[iz,iy,ix]=   g2Map[iz,iy,ix]  +   ss[g2name]
            nMap[iz,iy,ix] =   nMap[iz,iy,ix]   +   1.

    mask        =   (nMap>=0.1)
    g1Map[mask] =   g1Map[mask]/nMap[mask]
    g2Map[mask] =   g2Map[mask]/nMap[mask]
    pyfits.writeto(g1Fname,g1Map)
    pyfits.writeto(g2Fname,g2Map)
    pyfits.writeto(nFname,nMap)
    return

def main():
    # load data
    simDir  =   './'
    configName= 'config_lbd8_m9_z9.ini'
    srcFname    =   'src_'+configName.split('.')[0]+'.fits'
    srcFname    =   os.path.join(simDir,srcFname)
    parser      =   ConfigParser()
    parser.read(os.path.join(simDir,configName))
    sources     =   pyfits.getdata(srcFname)

    # pixelize
    pixelize(sources,parser)
    sparse3D    =   massmapSparsityTask(sources,parser)
    sparse3D.process()
    sparse3D.write()
    return


if __name__=='__main__':
    main()
