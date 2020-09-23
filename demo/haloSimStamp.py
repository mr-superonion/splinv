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

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

class haloSimStampBatchConfig(pexConfig.Config):
    noiDir  =   pexConfig.Field(dtype=str, default='sims/shapenoise_photoz-202003282257',
                doc = 'noise and redshift directory name')
    haloName=   pexConfig.Field(dtype=str, default='sims/haloCat-202009201646.csv',
                doc = 'halo catalog file name')
    configName  =   pexConfig.Field(dtype=str, default='config-pix96-nl15.ini',
                doc = 'configuration file name')
    outDir  =   pexConfig.Field(dtype=str, default='pix96-nl15/',
                doc = 'output directory')

    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    def validate(self):
        pexConfig.Config.validate(self)
        assert os.path.isdir(self.noiDir),\
            'Cannot find noise directory: %s' %self.noiDir
        assert os.path.isfile(self.haloName),\
            'Cannot find halo catalog file: %s' %self.haloName
        assert os.path.isfile(self.configName),\
            'Cannot find configure file: %s' %self.configName

class haloSimStampRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        # number of halos
        return [(ref, kwargs) for ref in range(64)]

def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)

class haloSimStampBatchTask(BatchPoolTask):
    ConfigClass = haloSimStampBatchConfig
    RunnerClass = haloSimStampRunner
    _DefaultName = "haloSimStampBatch"
    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))
    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)

    @abortOnError
    def runDataRef(self,Id):
        """
        @param id    group id
        """
        #Prepare the pool
        pool=   Pool("haloSimStamp")
        pool.cacheClear()
        pool.storeSet(noiDir=self.config.noiDir)
        pool.storeSet(haloName=self.config.haloName)
        pool.storeSet(configName=self.config.configName)
        pool.storeSet(outDir=self.config.outDir)
        ss  =   pyascii.read(self.config.haloName)[Id]
        pool.storeSet(ss=ss)
        self.pixelize_Sigma(ss)
        pool.map(self.process,range(100))
        return

    def pixelize_Sigma(self,ss):
        iz  =   ss['iz']
        im  =   ss['im']
        self.log.info('pixelization Sigma for halo iz: %d, im: %d' %(iz,im))
        # pixelaztion class
        parser  =   ConfigParser()
        parser.read(self.config.configName)
        gridInfo=   cartesianGrid3D(parser)

        # Lens Halo class
        M_200   =   10.**(ss['log10_M200'])
        conc    =   ss['conc']
        zh      =   ss['zh']
        halo    =   haloSim.nfw_lensTJ03(mass=M_200,conc=conc,redshift=zh,ra=0.,dec=0.)
        #halo=  haloSim.nfw_lensWB00(mass=M_200,conc=conc,redshift=zh,ra=0.,dec=0.)

        # Noise catalog
        noiName =   os.path.join(self.config.noiDir,'sim0.fits')
        noise   =   pyfits.getdata(noiName)

        # pixelize Sigma field
        sigmafname=   os.path.join(self.config.outdir,'pixsigma-%d%d.fits' %(iz,im))
        Sigma   =   halo.Sigma(noise['raR']*3600.,noise['decR']*3600.)
        ztmp    =   gridInfo.zcgrid[0]
        raname  =   'raR'
        decname =   'decR'
        pixSigma=   gridInfo.pixelize_data(noise[raname],noise[decname],ztmp,Sigma)[0]
        pyfits.writeto(Sigmafname,pixSigma,overwrite=True)
        return

    def process(self,cache,isim):
        """
        simulate shear field of halo and pixelize on postage-stamp
        @param cache    cache of the pool
        @param ss       halo source
        """

        ss  =   cache.ss
        iz  =   ss['iz']
        im  =   ss['im']
        self.log.info('simulating halo iz: %d, im: %d, realization: %d' %(iz,im,isim))

        # pixelaztion class
        parser  =   ConfigParser()
        parser.read(cache.configName)
        gridInfo=   cartesianGrid3D(parser)

        # Lens Halo class
        M_200   =   10.**(ss['log10_M200'])
        conc    =   ss['conc']
        zh      =   ss['zh']
        halo    =   haloSim.nfw_lensTJ03(mass=M_200,conc=conc,redshift=zh,ra=0.,dec=0.)
        #halo=  haloSim.nfw_lensWB00(mass=M_200,conc=conc,redshift=zh,ra=0.,dec=0.)

        # Noise catalog
        noiName =   os.path.join(cache.noiDir,'sim%d.fits' %isim)
        noise   =   pyfits.getdata(noiName)

        g1name  =   'g1R'   # random or hsc-like mask
        g2name  =   'g2R'
        raname  =   'raR'
        decname =   'decR'
        zname   =   'zbest' # best poz or true z

        deltaSigma= halo.DeltaSigmaComplex(noise['raR']*3600.,noise['decR']*3600.)
        lensKer =   halo.lensKernel(noise['ztrue']) # lensing kernel
        shear   =   deltaSigma*lensKer
        val     =   shear+noise['g1n']+noise['g2n']*1j
        g1g2    =   gridInfo.pixelize_data(noise[raname],noise[decname],noise[zname],val)

        g1fname     =   os.path.join(cache.outDir,'pixShearR-g1-%d%d-sim%d.fits' %(iz,im,isim))
        g2fname     =   os.path.join(cache.outDir,'pixShearR-g2-%d%d-sim%d.fits' %(iz,im,isim))
        pyfits.writeto(g1fname,g1g2.real,overwrite=True)
        pyfits.writeto(g2fname,g1g2.imag,overwrite=True)
        del g1g2
        del lensKer
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
