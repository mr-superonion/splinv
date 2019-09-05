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
import numpy as np
import astropy.io.fits as pyfits
from configparser import ConfigParser
from sparseBase import massmapSparsityTask

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import ArgumentParser, TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError


class sparse3D_s16aBatchConfig(pexConfig.Config):
    obsDir  =   pexConfig.Field(dtype=str, default='./s16a3D/3frames2Starlets/', doc="The output directory")
    lbd  =   pexConfig.Field(dtype=float, default=3.5, doc="The lambda of the sparsity algorithm")
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
        obsDir  =   self.config.obsDir
        lbd     =   self.config.lbd
        pool.storeSet(obsDir=obsDir)
        pool.storeSet(lbd=lbd)
        # Run the code with Pool
        pool.map(self.process,fieldList)
        return

    def process(self,cache,fieldName):
        obsDir      =   cache.obsDir
        lbd         =   cache.lbd
        self.log.info('processing field: %s' %fieldName)
        if fieldName != 'VVDS':
            self.log.info('stop processing field: %s' %fieldName)
            return

        outDir      =   os.path.join(obsDir,'lambda%.1f'%lbd)
        configName  =   'config_lbd%.1f_%s.ini' %(lbd,fieldName)
        configName  =   os.path.join(outDir,configName)

        inFname     =   './s16aPre/%s_RG.fits' %fieldName  
        sources     =   pyfits.getdata(inFname)

        parser      =   ConfigParser()
        parser.read(configName)
        sparse3D    =   massmapSparsityTask(sources,parser)
        sparse3D.process()
        sparse3D.write()
        
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
