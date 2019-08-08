#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 20082014 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
# python lib
import os
import time
import fitsio
import numpy as np
from configparser import SafeConfigParser

# lsst pipe basic
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.afw.image as afwImg
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord

# lsst Tasks
from lsst.pipe.base import ArgumentParser, TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError


class sparse3D_s16aBatchConfig(pexConfig.Config):
    lambdaR     =   pexConfig.Field(dtype=float, default=4, doc="regulation term lambda")
    pix_scale   =   pexConfig.Field(dtype=float, default=1./60, doc="pixel scale of the output mass map")
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)


class sparse3D_s16aRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [(ref, kwargs) for ref in range(1)] 

def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)

class sparse3D_s16aBatchTask(BatchPoolTask):
    ConfigClass = sparse3D_s16aBatchConfig
    RunnerClass = sparse3D_s16aRunner
    _DefaultName = "sparse3D_s16aBatch"

    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
    
    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))

    
    @abortOnError
    def run(self,Id):
        lbd     =   self.config.lambdaR
        glDir   =   './s16a3D/lambda_%s/' % lbd
        if not os.path.exists(glDir):
            os.mkdir(glDir)
        fieldList   =  np.load('/work/xiangchong.li/work/S16AFPFS/fieldInfo.npy').item().keys() 
        pool    =   Pool("sparse3D_s16aBatch")
        pool.cacheClear()
        pool.storeSet(lbd=lbd)
        pool.storeSet(pix_scale=self.config.pix_scale)
        # Run the code with Pool
        pool.map(self.process,fieldList)
        return

    def getFieldInfo(self,fieldName,pix_scale):
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


    def makeMaskMap(self,raCen,decCen,pix_scale,ngrid,catRG):
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


    def process(self,cache,fieldName):
        if fieldName!= 'VVDS':
            return
        self.log.info('processing field: %s' %fieldName)
        lbd         =   cache.lbd
        pix_scale   =   cache.pix_scale
        glDir   =   './s16a3D/lambda_%s/' % lbd
        if not os.path.exists(glDir):
            os.mkdir(glDir)
        inFname     =   './s16aPre2D/%s_RG.fits' %fieldName  
        fieldOut    =   self.getFieldInfo(fieldName,pix_scale)
        raCen,decCen,size,ngridX,ngridY,ngrid   =   fieldOut.values()
        mskFname    =   os.path.join(glDir,'msk_%s.fits'  %(fieldName))
        if not os.path.exists(mskFname):
            catRG   =   fitsio.read(inFname)
            mskMap  =   self.makeMaskMap(raCen,decCen,pix_scale,ngrid,catRG)
            fitsio.write(mskFname,mskMap)
        outFname    =   'kappaMap_GL_E_%s.fits' %(fieldName)
        outFname    =   os.path.join(glDir,outFname)
        if not os.path.exists(outFname):
            configName  =   '/work/xiangchong.li/superonionGW/code/kappaMap_Private/config/s16a3DConfig.ini'
            configName2 =   os.path.join(glDir,'s16aConfig_%s.ini' %fieldName)
            parser = SafeConfigParser()
            parser.read(configName)
            parser.set('field','pixel_size','%s' % pix_scale)
            parser.set('field','npix','%s' % ngrid)
            parser.set('field','nlp','1')
            parser.set('survey','size','%s' % size)
            parser.set('survey','center_ra','%s' % raCen)
            parser.set('survey','center_dec','%s' % decCen)
            parser.set('survey','zsig_min','0')
            parser.set('survey','zsig_max','0')
            parser.set('parameters','lambda','%s' % lbd)
            parser.set('parameters','nrandom','100')
            parser.set('parameters','niter','500')
            parser.set('parameters','nreweights','0')
            with open(configName2, 'w') as configfile:
                parser.write(configfile)
            binDir      =   '/work/xiangchong.li/superonionGW/packages/Glimpse/build'
            os.system('%s/glimpse %s %s %s' %(binDir,configName2,inFname,outFname))
        else:
            self.log.info('already have output files')
        if os.path.exists(mskFname) and os.path.exists(outFname) :
            mskMap  =   fitsio.read(mskFname)
            outMap  =   fitsio.read(outFname)
            shiftX  =   int((ngrid-ngridX)//2)
            shiftY  =   int((ngrid-ngridY)//2)
            fieldOut.update({'pix_scale':pix_scale})
            if len(outMap.shape)==2:
                outMap  =   outMap*mskMap
                fitsio.write(outFname+'2',data=outMap[shiftY:shiftY+ngridY,shiftX:shiftX+ngridX],header=fieldOut,clobber=True)
            if len(outMap.shape)==3:
                for iz in range(outMap.shape[0]):
                    outMap[iz,:,:]  =   outMap[iz,:,:]*mskMap
                fitsio.write(outFname+'2',data=outMap[:,shiftY:shiftY+ngridY,shiftX:shiftX+ngridX],header=fieldOut,clobber=True)
        return
    
        
    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = ArgumentParser(name=cls._DefaultName)
        return parser
    
    @classmethod
    def writeConfig(self, butler, clobber=False, doBackup=False):
        pass
    
    def writeSchemas(self, butler, clobber=False, doBackup=False):
        pass

    def writeMetadata(self, dataRef):
        pass

    def writeEupsVersions(self, butler, clobber=False, doBackup=False):
        pass
    
    def _getConfigName(self):
        return None

    def _getEupsVersionsName(self):
        return None

    def _getMetadataName(self):
        return None
