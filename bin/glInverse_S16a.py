#!/usr/bin/env python
import os
import fitsio
import numpy as np
from configparser import SafeConfigParser

def getFieldInfo(fieldName,pix_scale):
    fieldInfo=  np.load('/work/xiangchong.li/work/S16AFPFS/fieldInfo.npy').item()[fieldName]
    raMin   =   fieldInfo['raMin']
    raMax   =   fieldInfo['raMax']
    raCen   =   (raMin+raMax)/2.-pix_scale/2.
    decMin  =   fieldInfo['decMin']
    decMax  =   fieldInfo['decMax']
    decCen  =   (decMin+decMax)/2.-pix_scale/2.
    sizeX   =   raMax-raMin
    sizeY   =   decMax-decMin
    ngridX  =   np.int(sizeX/pix_scale)
    ngridY  =   np.int(sizeY/pix_scale)
    if ngridX%2  !=  0:
        ngridX  =   ngridX+1
    if ngridY%2  !=  0:
        ngridY  =   ngridY+1
    ngrid   =   max(ngridX,ngridY)
    size    =   pix_scale*ngrid
    outDict =   {'raCent':raCen,'decCent':decCen,'size':size,'ngridX':ngridX,'ngridY':ngridY,'ngrid':ngrid}
    return outDict 


def makeMaskMap(raCen,decCen,pix_scale,ngrid,catRG):
    xMin    =   raCen   -   pix_scale*ngrid/2
    yMin    =   decCen  -   pix_scale*ngrid/2
    ra      =   catRG['ra']
    dec     =   catRG['dec']
    maskMap =   np.zeros((ngrid,ngrid))
    for nra,ndec in zip(ra,dec):
        raBin   =   int((nra-xMin)/pix_scale)
        decBin  =   int((ndec-yMin)/pix_scale)
        maskMap[decBin,raBin]  =   maskMap[decBin,raBin]+1
    maskMap =   (maskMap>=2).astype(int)
    return maskMap
    
def main():
    lbd         =   4
    fieldName   =   'VVDS'
    process(fieldName)
    return

def process(fieldName):
    glDir       =   './s16a/lambda%d/' % lbd
    if not os.path.exists(glDir):
        os.mkdir(glDir)
    inFname     =   './s16aPre/%s_RG.fits' %fieldName  
    pix_scale   =   0.01667
    fieldOut    =   getFieldInfo(fieldName,pix_scale)
    raCen,decCen,size,ngridX,ngridY,ngrid   =   fieldOut.values()
    mskFname    =   os.path.join(glDir,'msk_%s.fits'  %(fieldName))
    if not os.path.exists(mskFname):
        catRG   =   fitsio.read(inFname)
        mskMap  =   makeMaskMap(raCen,decCen,pix_scale,ngrid,catRG)
        fitsio.write(mskFname,mskMap)

    outFname    =   'kappaMap_GL_E_%s.fits' %(fieldName)
    outFname    =   os.path.join(glDir,outFname)
    if not os.path.exists(outFname):
        configName  =   '/work/xiangchong.li/superonionGW/code/kappaMap_Private/config/s16aConfig.ini'
        parser = SafeConfigParser()
        parser.read(configName)
        parser.set('field','pixel_size','%s' % pix_scale)
        parser.set('field','npix','%s' % ngrid)
        parser.set('survey','size','%s' % size)
        parser.set('survey','center_ra','%s' % raCen)
        parser.set('survey','center_dec','%s' % decCen)
        parser.set('parameters','lambda','%s' % lbd)
        with open(configName, 'w') as configfile:
            parser.write(configfile)
        binDir      =   '/work/xiangchong.li/superonionGW/packages/Glimpse/build'
        os.system('%s/glimpse %s %s %s' %(binDir,configName,inFname,outFname))

    if os.path.exists(mskFname) and os.path.exists(outFname) :
        mskMap  =   fitsio.read(mskFname)
        outMap  =   fitsio.read(outFname)
        outMap  =   outMap*mskMap
        shiftX  =   int((ngrid-ngridX)//2)
        shiftY  =   int((ngrid-ngridY)//2)
        fieldOut.update({'pix_scale':pix_scale})
        fitsio.write(outFname+'2',data=outMap[shiftY:shiftY+ngridY,shiftX:shiftX+ngridX],header=fieldOut)
    return

if __name__ == '__main__':
    main()
