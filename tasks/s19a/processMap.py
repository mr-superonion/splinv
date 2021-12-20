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
from sparseBase import massmapSparsityTaskNew

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

class processMapBatchConfig(pexConfig.Config):
    configName  =   pexConfig.Field(dtype=str, default='sim/config-obs-nl12-pc.ini',
                    doc = 'configuration file name')
    pixDir      =   pexConfig.Field(dtype=str, default='test4/pixes-mock/',
                    doc = 'pixelization directory name')
    outDir      =   pexConfig.Field(dtype=str, default='test4/out-mock',
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
        #return [(ref, kwargs) for ref in utility.field_names]
        return [(ref, kwargs) for ref in range(1)]
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
        pool.storeSet(outDir=self.config.outDir)
        nrun    =   100
        pool.map(self.process,range(50,200))
        return

    def prepareParser(self,cache,irun):
        # configuration
        parser  =   ConfigParser()
        parser.read(cache.configName)

        g1fname =   os.path.join(cache.pixDir,'g1Map-dempz-%d.fits' %irun)
        g2fname =   os.path.join(cache.pixDir,'g2Map-dempz-%d.fits' %irun)
        stdfname=   os.path.join(cache.outDir,'stdMap-dempz.fits')
        lenskerfname=os.path.join(cache.outDir,'lensker-dempz-10bins.fits')
        dictfname=  'sim/prior/haloBasis-obs-nl12-m146_152.fits'
        assert os.path.isfile(g1fname)
        assert os.path.isfile(g2fname)
        assert os.path.isfile(stdfname)
        assert os.path.isfile(lenskerfname)
        assert os.path.isfile(dictfname)

        parser.set('prepare','g1fname',g1fname)
        parser.set('prepare','g2fname',g2fname)
        parser.set('prepare','sigmafname',stdfname)
        parser.set('prepare','lkfname',lenskerfname)
        parser.set('lensZ','atomFname',dictfname)
        __tmp   =   pyfits.getdata(stdfname)
        nz,ny,nx=   __tmp.shape
        parser.set('transPlane','ny','%d' %ny)
        parser.set('transPlane','nx','%d' %nx)

        parser.set('sparse','lbd','1.')
        parser.set('sparse','lcd','0.6')

        parser.set('sparse','nframe','1' )
        parser.set('sparse','mu','4e-4')

        return parser

    def process(self,cache,irun):
        """
        simulate shear field of halo and pixelize on postage-stamp
        Parameters:
        @param cache:       cache of the pool
        @param irun:        simulation id
        """

        parser   =  self.prepareParser(cache,irun)
        outfnameP1=  os.path.join(cache.outDir,'alphaRW-enet-%d.fits' %irun)
        outfnameP2=  os.path.join(cache.outDir,'deltaR-enet-%d.fits' %irun)
        outfname1=  os.path.join(cache.outDir,'alphaRW-alasso-%d.fits' %irun)
        outfname2=  os.path.join(cache.outDir,'deltaR-alasso-%d.fits' %irun)
        if not os.path.isfile(outfnameP2):
            sparse3D    =   massmapSparsityTaskNew(parser)
            sparse3D.clean_outcome()
            sparse3D.lbd=   0.
            sparse3D.lcd=   1.
            sparse3D.nonNeg=False
            sparse3D.fista_gradient_descent(1500)
            pyfits.writeto(outfnameP1,sparse3D.alphaR,overwrite=True)
            sparse3D.reconstruct()
            pyfits.writeto(outfnameP2,sparse3D.deltaR,overwrite=True)
            del sparse3D
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
