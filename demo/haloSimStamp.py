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
    obsDir  =   pexConfig.Field(dtype=str, default='HSC-obs/20200328/',
                doc = 'obs directory name')
    haloName=   pexConfig.Field(dtype=str,
                default='planck-cosmo/nfw-halos/haloCat-202010032144.csv',
                doc = 'halo catalog file name')
    configName  =   pexConfig.Field(dtype=str,
                default='planck-cosmo/config-pix96-nl20.ini',
                doc = 'configuration file name')
    outDir  =   pexConfig.Field(dtype=str,
                default='planck-cosmo/pix96-ns10/',
                doc = 'output directory')
    """
    outDir  =   pexConfig.Field(dtype=str,
                default='HSC-obs/20200328/pixes/',
                doc = 'output directory')
    """

    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    def validate(self):
        pexConfig.Config.validate(self)
        assert os.path.isdir(self.obsDir),\
            'Cannot find observation directory: %s' %self.obsDir
        assert os.path.isfile(self.haloName),\
            'Cannot find halo catalog file: %s' %self.haloName
        assert os.path.isfile(self.configName),\
            'Cannot find configure file: %s' %self.configName

class haloSimStampRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        # number of halos
        #idRange=range(64)
        idRange=[10000000]
        return [(ref, kwargs) for ref in idRange]

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
        pool.storeSet(obsDir=self.config.obsDir)
        pool.storeSet(haloName=self.config.haloName)
        pool.storeSet(configName=self.config.configName)
        if Id<5e4:
            # halo field
            ss  =   pyascii.read(self.config.haloName)[Id]
            iz  =   ss['iz']
            im  =   ss['im']
            self.pixelize_Sigma(ss)
        else:
            # Noise field
            ss  =   None
            iz  =   800
            im  =   800
        pool.storeSet(ss=ss)
        outDirH =   os.path.join(self.config.outDir,'halo%d%d'%(iz,im))
        if not os.path.isdir(outDirH):
            os.mkdir(outDirH)
        pool.storeSet(outDirH=outDirH)
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
        omega_m =   parser.getfloat('cosmology','omega_m')

        # Lens Halo class
        M_200   =   10.**(ss['log10_M200'])
        conc    =   ss['conc']
        zh      =   ss['zh']
        # TJ03 halo
        halo    =   haloSim.nfw_lensTJ03(mass=M_200,conc=conc,\
                    redshift=zh,ra=0.,dec=0.,omega_m=omega_m)
        """
        # WB00 halo
        halo    =   haloSim.nfw_lensWB00(mass=M_200,conc=conc,\
                    redshift=zh,ra=0.,dec=0.,omega_m=omega_m)
        """

        # Noise catalog
        obsName =   os.path.join(self.config.obsDir,'cats','sim0.fits')
        obs     =   pyfits.getdata(obsName)

        # pixelize Sigma field
        Sigmafname= os.path.join(self.config.outDir,'pixsigma-%d%d.fits' %(iz,im))
        Sigma   =   halo.Sigma(obs['raR']*3600.,obs['decR']*3600.)
        ztmp    =   gridInfo.zcgrid[0]
        raname  =   'raR'
        decname =   'decR'
        pixSigma=   gridInfo.pixelize_data(obs[raname],obs[decname],ztmp,Sigma)[0]
        pyfits.writeto(Sigmafname,pixSigma,overwrite=True)
        return

    def process(self,cache,isim):
        """
        simulate shear field of halo and pixelize on postage-stamp
        @param cache    cache of the pool
        @param ss       halo source
        """

        # Noise catalog
        obsName =   os.path.join(cache.obsDir,'cats','sim%d.fits' %isim)
        obs     =   pyfits.getdata(obsName)

        g1name  =   'g1R'   # random or hsc-like mask
        g2name  =   'g2R'
        raname  =   'raR'
        decname =   'decR'
        zname   =   'zbest' # best poz or true z

        parser  =   ConfigParser()
        parser.read(cache.configName)
        gridInfo=   cartesianGrid3D(parser)

        ss  =   cache.ss
        if ss is not None:
            iz  =   ss['iz']
            im  =   ss['im']
            self.log.info('simulating halo iz: %d, im: %d, realization: %d' %(iz,im,isim))

            # pixelaztion class
            omega_m =   parser.getfloat('cosmology','omega_m')

            # Lens Halo class
            M_200   =   10.**(ss['log10_M200'])
            conc    =   ss['conc']
            zh      =   ss['zh']
            # TJ03 halo
            halo    =   haloSim.nfw_lensTJ03(mass=M_200,conc=conc,\
                        redshift=zh,ra=0.,dec=0.,omega_m=omega_m)
            """
            # WB00 halo
            halo    =   haloSim.nfw_lensWB00(mass=M_200,conc=conc,\
                        redshift=zh,ra=0.,dec=0.,omega_m=omega_m)
            """
            deltaSigma= halo.DeltaSigmaComplex(obs['raR']*3600.,obs['decR']*3600.)
            lensKer =   halo.lensKernel(obs['ztrue']) # lensing kernel
            shear   =   deltaSigma*lensKer
        else:
            shear   =   0.
            iz      =   800
            im      =   800

        val     =   obs['g1n']+obs['g2n']*1j+shear
        g1g2    =   gridInfo.pixelize_data(obs[raname],obs[decname],obs[zname],val)

        pnm     =   '%d%d-sim%d' %(iz,im,isim)
        g1fname     =   os.path.join(cache.outDirH,'pixShearR-g1-%s.fits' %pnm)
        g2fname     =   os.path.join(cache.outDirH,'pixShearR-g2-%s.fits' %pnm)
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
