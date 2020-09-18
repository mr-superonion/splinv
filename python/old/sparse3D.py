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
import fitsio
import galsim
import numpy as np
import astropy.io.fits as pyfits
from astropy.table import Table
from configparser import ConfigParser
from sparseBase import massmapSparsityTask

# lsst Tasks
import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.afw.image as afwImg
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
import lsst.meas.algorithms as meaAlg

from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

class sparse3DConfig(pexConfig.Config):
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    def validate(self):
        pexConfig.Config.validate(self)

class sparse3DTask(pipeBase.CmdLineTask):
    _DefaultName = "sparse3D"
    ConfigClass = sparse3DConfig
    def __init__(self,**kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)

    @pipeBase.timeMethod
    def run(self,configName,simDir):
        configName  =   configName[:-1]
        self.log.info(configName)
        srcFname    =   'src_'+configName.split('.')[0]+'.fits'
        srcFname    =   os.path.join(simDir,srcFname)
        parser      =   ConfigParser()
        parser.read(os.path.join(simDir,configName))
        z_id        =   parser.getint('lens','z_id')             #redshift
        m_id        =   parser.getint('lens','m_id')             #arcsec
        M200        =   parser.getfloat('lens','m_200')          #redshift
        z_cl        =   parser.getfloat('lens','z_cl')           #arcsec
        lbd         =   parser.getfloat('sparse','lbd')
        outFname    =   os.path.join(simDir,'kappaMap_3D_lbd%s_m%d_z%d.fits'%(lbd,m_id,z_id))
        if os.path.exists(outFname):
           massMap  =   pyfits.getdata(outFname)
        else:
            sources     =   pyfits.getdata(srcFname)
            sparse3D    =   massmapSparsityTask(sources,parser)
            sparse3D.process()
            massMap     =   sparse3D.deltaR.real
            pyfits.writeto(outFname,massMap,overwrite=True)
        zyx         =   np.argmax(massMap)
        iz,iy,ix    =   np.unravel_index(zyx,massMap.shape)
        return (M200,z_cl,ix,iy,iz)

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        doBatch = kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        return parser
    def writeConfig(self, butler, clobber=False, doBackup=False):
        pass
    def writeSchemas(self, butler, clobber=False, doBackup=False):
        pass
    def writeMetadata(self, dataRef):
        pass
    def writeEupsVersions(self, butler, clobber=False, doBackup=False):
        pass

class sparse3DBatchConfig(pexConfig.Config):
    sparse3D = pexConfig.ConfigurableField(
        target = sparse3DTask,
        doc = "sparse3D task to run on multiple cores"
    )
    simDir  =   pexConfig.Field(dtype=str, default='simulation9',
                doc = 'directory to store exposures')
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    def validate(self):
        pexConfig.Config.validate(self)
        if not os.path.exists(self.simDir):
            self.log.info('Do not have %s' %self.simDir)

class sparse3DRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [(ref, kwargs) for ref in range(1)]
def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)
class sparse3DBatchTask(BatchPoolTask):
    ConfigClass = sparse3DBatchConfig
    RunnerClass = sparse3DRunner
    _DefaultName = "sparse3DBatch"
    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))
    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        self.makeSubtask("sparse3D")

    @abortOnError
    def run(self,Id):
        #Prepare the pool
        pool    =   Pool("sparse3D")
        pool.cacheClear()
        simDir  =   self.config.simDir
        pool.storeSet(simDir=simDir)
        configList= os.popen('ls %s/ |grep ini |grep -v inverse'%simDir).readlines()
        rows    =   pool.map(self.process,configList)
        tabOut  =   Table(rows=rows, names=('M200','z_cl','x','y','z'))
        tabOut.write('detection_lbd8.csv',overwrite=True)
        self.log.info('finish group %d'%(Id))
        return
    def process(self,cache,ifield):
        return self.sparse3D.run(ifield,cache.simDir)

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
