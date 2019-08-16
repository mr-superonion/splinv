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
from sparseBase import massmap_sparsity_3D_2

# lsst pipe basic
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

# lsst Tasks
from lsst.pipe.base import ArgumentParser, TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

class sparseMockCatBatchConfig(pexConfig.Config):
    outDir  =   pexConfig.Field(dtype=str, default='./s16a3D2/', doc="The output directory")
    nSim    =   pexConfig.Field(dtype=int, default=100, doc="number of realization")
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)

class sparseMockCatRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        configList= os.popen('ls ./s16a3D2/ |grep ini |grep -v inverse').readlines()
        return [(ref, kwargs) for ref in configList]

def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)

class sparseMockCatBatchTask(BatchPoolTask):
    ConfigClass = sparseMockCatBatchConfig
    RunnerClass = sparseMockCatRunner
    _DefaultName = "sparseMockCatBatch"

    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
    
    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))

    
    @abortOnError
    def run(self,configName):
        outDir      =   self.config.outDir
        configName  =   configName[:-1]
        configName2 =   configName.split('.')[0][7:] 
        fieldName   =   configName2.split('_')[1]
        self.log.info('processing field: %s' %fieldName)
        outFname    =   'mock_%s.npy' %(configName2)
        outFname    =   os.path.join(outDir,outFname)
        pool        =   Pool("sparseMockCatBatch")
        pool.cacheClear()
        parser      =   ConfigParser()
        parser.read(os.path.join(outDir,configName))
        pool.storeSet(parser=parser)
        pool.storeSet(fieldName=fieldName)
        # Run the code with Pool
        nSim        =   self.config.nSim
        shearAll    =   pool.map(self.process,range(nSim))
        self.log.info('writing outcome for field: %s' %fieldName)
        shearAll    =   np.array(shearAll)
        np.save(outFname,shearAll)
        return

    def process(self,cache,isim):
        self.log.info('processing simulation: %d' %isim)
        parser      =   cache.parser
        fieldName   =   cache.fieldName
        simSrcName  =   './s16aPre2D/%s_RG_mock.fits' %(fieldName)
        simSrc      =   pyfits.getdata(simSrcName)
        #transverse plane
        if parser.has_option('transPlane','raname'):
            raname =   parser.get('transPlane','raname')
        else:
            raname =   'ra'
        if parser.has_option('transPlane','decname'):
            decname=   parser.get('transPlane','decname')
        else:
            decname=   'dec'
        xMin    =   parser.getfloat('transPlane','xMin')
        yMin    =   parser.getfloat('transPlane','yMin')
        scale   =   parser.getfloat('transPlane','scale')
        ny      =   parser.getint('transPlane'  ,'ny')
        nx      =   parser.getint('transPlane'  ,'nx')
        #source z axis
        if parser.has_option('sourceZ','zname'):
            zname      =   parser.get('sourceZ','zname')
        else:
            zname      =   'z'
        zname   =   zname+'_%d'  %isim
        zMin    =   parser.getfloat('sourceZ','zMin')
        zscale  =   parser.getfloat('sourceZ','zscale')
        nz      =   parser.getint('sourceZ','nz')
        shapeS  =   (nz,ny,nx)   
        nSim    =   np.zeros(shapeS)  
        g1Sim   =   np.zeros(shapeS)
        g2Sim   =   np.zeros(shapeS)
        for ss in simSrc:
            ix  =   int((ss[raname]-xMin)//scale)
            iy  =   int((ss[decname]-yMin)//scale)
            iz  =   int((ss[zname]-zMin)//zscale)
            if iz>=0 and iz<nz:
                g1Sim[iz,iy,ix]=   g1Sim[iz,iy,ix]+ss['g1_%d'%isim]
                g2Sim[iz,iy,ix]=   g2Sim[iz,iy,ix]+ss['g2_%d'%isim]
                nSim[iz,iy,ix] =   nSim[iz,iy,ix]+1.
        mask        =   (nSim>=0.1)
        g1Sim[mask] =   g1Sim[mask]/nSim[mask]
        g2Sim[mask] =   g2Sim[mask]/nSim[mask]
        shearSim    =   g1Sim+np.complex128(1j)*g2Sim
        return shearSim

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
