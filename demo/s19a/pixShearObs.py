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
import haloSim
import numpy as np
import astropy.io.fits as pyfits
import astropy.io.ascii as pyascii
from pixel3D import cartesianGrid3D
from configparser import ConfigParser
import sim_analysis_utilities as utility

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

class pixShearObsBatchConfig(pexConfig.Config):
    obsDir  =   pexConfig.Field(dtype=str, default='HSC-obs/20200328/',
                doc = 'obs directory name')
    configName  =   pexConfig.Field(dtype=str,
                default='planck-cosmo/config-pix96-nl20.ini',
                doc = 'configuration file name')
    outDir  =   pexConfig.Field(dtype=str,
                default='planck-cosmo/pix96-ns10/',
                doc = 'output directory')

    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    def validate(self):
        pexConfig.Config.validate(self)
        assert os.path.isdir(self.obsDir),\
            'Cannot find observation directory: %s' %self.obsDir
        assert os.path.isfile(self.configName),\
            'Cannot find configure file: %s' %self.configName

class pixShearObsRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        # number of halos
        return [(ref, kwargs) for ref in utility.field_names]

def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)

class pixShearObsBatchTask(BatchPoolTask):
    ConfigClass = pixShearObsBatchConfig
    RunnerClass = pixShearObsRunner
    _DefaultName = "pixShearObsBatch"
    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))
    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)

    @abortOnError
    def runDataRef(self,fieldname):
        """
        @param id    group id
        """
        #Prepare the pool
        pool    =   Pool("pixShearObs")
        pool.cacheClear()
        pool.storeSet(obsDir=self.config.obsDir)
        pool.storeSet(configName=self.config.configName)
        pool.storeSet(fieldname=fieldname)
        outDir =   os.path.join(self.config.outDir,fieldname)
        if not os.path.isdir(outDirH):
            os.system('mkdir -p %s' %outDir)
        pool.storeSet(outDirH=outDir)
        nrun    =   1
        pool.map(self.process,range(nrun))
        return

    def process(self,cache,fieldname):
        """
        simulate shear field of halo and pixelize on postage-stamp
        Parameters:
        @param cache        cache of the pool
        @param fieldname    field name of HSC observation
        """

        # Noise catalog
        obsName =   os.path.join(cache.obsDir,'%s.fits' %fieldname)
        obs     =   pyfits.getdata(obsName)

        parser  =   ConfigParser()
        parser.read(cache.configName)
        gridInfo=   cartesianGrid3D(parser)

        self.log.info('processing field %s' %fieldname)

        # pixelaztion class
        val     =   obs['g1n']+obs['g2n']*1j+shear
        g1g2    =   gridInfo.pixelize_data(obs[raname],obs[decname],obs[zname],val)

        pnm     =   '%s' %(fieldname)
        g1fname =   os.path.join(cache.outDirH,'pixShearR-g1-%s.fits' %pnm)
        g2fname =   os.path.join(cache.outDirH,'pixShearR-g2-%s.fits' %pnm)
        pyfits.writeto(g1fname,g1g2.real,overwrite=True)
        pyfits.writeto(g2fname,g1g2.imag,overwrite=True)
        del g1g2
        del shear
        del val
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
