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
        fieldList   =  np.load('/work/xiangchong.li/work/S16AFPFS/fieldInfo.npy').item().keys() 
        pool    =   Pool("sparse3D_s16aBatch")
        pool.cacheClear()
        #pool.storeSet()
        #pool.storeSet()
        # Run the code with Pool
        pool.map(self.process,fieldList)
        return



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
        
        else:
            self.log.info('already have output files')
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
