#!/usr/bin/env python
import os
import numpy as np
from configparser import ConfigParser

def getFieldInfo(fieldName,pix_scale):
    fieldInfo=  np.load('/work/xiangchong.li/work/S16AFPFS/fieldInfo.npy').item()[fieldName]
    raMin   =   fieldInfo['raMin']
    raMax   =   fieldInfo['raMax']
    decMin  =   fieldInfo['decMin']
    decMax  =   fieldInfo['decMax']
    sizeX   =   raMax-raMin
    sizeY   =   decMax-decMin
    centX   =   (raMax+raMin)/2.
    centY   =   (decMax+decMin)/2.
    ngridX  =   np.int(sizeX/pix_scale)
    ngridY  =   np.int(sizeY/pix_scale)
    if ngridX%2  !=  0:
        ngridX  =   ngridX+1
    if ngridY%2  !=  0:
        ngridY  =   ngridY+1
    ngrid   =   max(ngridX,ngridY)
    raMin   =   centX-ngrid/2*pix_scale
    decMin  =   centY-ngrid/2*pix_scale
    return  raMin,decMin,ngrid

def addInfoSparse(parser,lbd,fieldName):
    #sparse
    doDebug=    'no'
    nframe =    1
    nMax   =    1
    maxR   =    2

    #sparse
    parser['sparse']={  'doDebug':'%s'%doDebug,
                        'lbd'   :'%s' %lbd,
                        'nframe':'%s' %nframe,
                        'nMax'  :'%s' %nMax,
                        'maxR'  :'%s' %maxR
                        }
    #transverse plane
    raname      =   'ra'
    decname     =   'dec'
    unit        =   'degree'
    scale       =   0.05    #(degree/pix)
    xMin,yMin,nxy=  getFieldInfo(fieldName,scale)
    ny          =   nxy     #pixels
    nx          =   nxy     #pixels

    parser['transPlane']={'raname':'%s'  %raname,
                         'decname':'%s'  %decname,  
                         'unit'   :'%s'  %unit,
                         'xMin'   :'%s'  %xMin,
                         'yMin'   :'%s'  %yMin,
                         'scale'  :'%s'  %scale,
                         'ny'     :'%s'  %ny,
                         'nx'     :'%s'  %nx}

    #lens z axis
    zname       =   'mlz_photoz_best'
    nlp         =   20
    zlMin       =   0.01
    zlscale     =   0.05
    parser['lensZ']={'zname'    :'%s'  %zname,
                     'zlMin'    :'%s'  %zlMin,
                     'zlscale'  :'%s'  %zlscale,
                     'nlp'      :'%s'  %nlp}

    #source z axis
    nz          =   8
    if nz!=1:
        zMin    =   0.05
        zscale  =   0.25
    else:
        zMin    =   0.
        zscale  =   4.
    parser['sourceZ']={ 'zMin'  :   '%s'  %zMin,
                        'zscale':   '%s'  %zscale,
                        'nz'    :   '%s'  %nz}
    return parser



if __name__=='__main__':
    lbd     =   8
    obsDir  =   's16a3D2'
    if not os.path.exists(obsDir):
        os.mkdir(obsDir)
    fdNames =  np.load('/work/xiangchong.li/work/S16AFPFS/fieldInfo.npy').item().keys()
    for fieldName in fdNames:
        configName  =   os.path.join(obsDir,'config_lbd%d_%s.ini' %(lbd,fieldName))
        parser  =   ConfigParser()
        parser  =   addInfoSparse(parser,lbd,fieldName)
        #file
        parser['file']= { 'root'    :'%s'%obsDir,
                          'fieldN'  :'%s'%fieldName
                          }
        with open(configName, 'w') as configfile:
            parser.write(configfile)
