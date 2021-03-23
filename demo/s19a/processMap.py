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
import gc
import numpy as np
import astropy.io.fits as pyfits
from configparser import ConfigParser
import sim_analysis_utilities as utility
from sparseBase import massmapSparsityTaskNew

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

class processMapBatchConfig(pexConfig.Config):
    configName  =   pexConfig.Field(dtype=str, default='config-nl10.ini',
                    doc = 'configuration file name')
    pixDir      =   pexConfig.Field(dtype=str, default='test/mock/',
                    doc = 'pixelization directory name')
    outDir      =   pexConfig.Field(dtype=str, default='planck-cosmo/pix96-ns10/',
                    doc = 'output directory name')
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    def validate(self):
        pexConfig.Config.validate(self)
        assert os.path.isfile(self.configName),\
            'Cannot find configure file: %s' %self.configName
        assert os.path.isdir(self.pixDir),\
            'Cannot find pixelization directory: %s' %self.pixDir

class processMapRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        # number of halos
        return [(ref, kwargs) for ref in utility.field_names]
def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)

class processMapBatchTask(BatchPoolTask):
    ConfigClass = processMapBatchConfig
    RunnerClass = processMapRunner
    _DefaultName = "processMapBatch"
    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))
    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)

    @abortOnError
    def runDataRef(self,fieldname):
        """
        @param id:    group id
        """
        #Prepare the pool
        pool    =   Pool("processMap")
        pool.cacheClear()
        pool.storeSet(configName=self.config.configName)
        pool.storeSet(pixDir=self.config.pixDir)
        pool.storeSet(fieldname=fieldname)
        # outDir  =   os.path.join(self.config.outDir,fieldname)
        # if not os.path.isdir(outDirH):
        #     os.system('mkdir -p %s' %outDir)
        pool.storeSet(outDirH=outDir)
        nrun    =   1
        pool.map(self.process,range(nrun))
        return

    def process(self,cache,irun):
        """
        simulate shear field of halo and pixelize on postage-stamp
        Parameters:
        @param cache:       cache of the pool
        @param irun:        field name of HSC observation
        """
        g1fname =   os.path.join(cache.pixDir,'g1Map-dempz-%d.fits' %irun)
        g2fname =   os.path.join(cache.pixDir,'g2Map-dempz-%d.fits' %irun)
        stdfname=   os.path.join(cache.pixDir,'stdMap-dempz.fits')
        lenskerfname=os.path.join(cache.outDirH,'lensker-dempz-10bins.fits')

        parser  =   ConfigParser()
        parser.read(cache.configName)

        gc.collect()
        return

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
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
