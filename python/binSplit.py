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

# lsst Tasks
import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import ArgumentParser, TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

class binSplitBatchConfig(pexConfig.Config):
    nSim    =   pexConfig.Field(dtype=int, default=100, doc="number of realization")
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)

class binSplitRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        configDir    =  parsedCmd.configDir 
        ##!NOTE: only try on VVDS
        configList= os.popen('ls %s* |grep VVDS.ini ' %configDir).readlines()
        return [(ref, kwargs) for ref in configList]

def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)

class binSplitBatchTask(BatchPoolTask):
    ConfigClass = binSplitBatchConfig
    RunnerClass = binSplitRunner
    _DefaultName = "binSplitBatch"

    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))
    
    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)

    @abortOnError
    def runDataRef(self,configName):
        configName  =   configName[:-1]
        parser      =   ConfigParser()
        parser.read(configName)
        fieldName   =   parser.get('file','fieldN')
        pixDir      =   parser.get('file','pixDir')
        ##!NOTE: the mock grid file is saved under pix directroy 
        ##since changing lbd does not require new pixelation 
        ##changing nframe only change sigmaA and 
        ##also does not need new pixelation 
        ##changing pix_size change shearAll and sigmaA
        ##which need new pixelation
        self.log.info('processing field: %s' %fieldName)
        # pixelize the mock catalogs
        outFname    =   'mock_RG_grid_%s.npy' %(fieldName)
        outFname    =   os.path.join(pixDir,outFname)
        if not os.path.exists(outFname):
            pool    =   Pool("binSplitBatch")
            pool.cacheClear()
            pool.storeSet(parser=parser)
            # Run the code with Pool
            nSim    =   self.config.nSim
            shearAll=   pool.map(self.process,range(nSim))
            self.log.info('writing outcome for field: %s' %fieldName)
            shearAll=   np.array(shearAll)
            np.save(outFname,shearAll)
        # pixelize the true catalog
        g1Fname     =   os.path.join(pixDir,'g1Map_%s.fits' %fieldName)
        g2Fname     =   os.path.join(pixDir,'g2Map_%s.fits' %fieldName)
        nFname      =   os.path.join(pixDir,'nMap_%s.fits'  %fieldName)
        if parser.has_option('sourceZ','zname'):
            zname   =   parser.get('sourceZ','zname')
        else:
            zname   =   'z'
        g1name  =   'g1'
        g2name  =   'g2'

        catSrcName  =   './s16aPre/%s_RG.fits' %(fieldName)
        catSrc      =   pyfits.getdata(catSrcName)
        g1Map,g2Map,nMap=   self.pixelize(catSrc,zname,g1name,g2name,parser)
        pyfits.writeto(g1Fname,g1Map)
        pyfits.writeto(g2Fname,g2Map)
        pyfits.writeto(nFname,nMap)
        return

    def pixelize(self,simSrc,zname,g1name,g2name,parser):
        #transverse plane
        if parser.has_option('transPlane','raname'):
            raname  =   parser.get('transPlane','raname')
        else:
            raname  =   'ra'
        if parser.has_option('transPlane','decname'):
            decname=   parser.get('transPlane','decname')
        else:
            decname =   'dec'
        xMin    =   parser.getfloat('transPlane','xMin')
        yMin    =   parser.getfloat('transPlane','yMin')
        scale   =   parser.getfloat('transPlane','scale')
        ny      =   parser.getint('transPlane'  ,'ny')
        nx      =   parser.getint('transPlane'  ,'nx')
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
                g1Sim[iz,iy,ix]=   g1Sim[iz,iy,ix]  +   ss[g1name]
                g2Sim[iz,iy,ix]=   g2Sim[iz,iy,ix]  +   ss[g2name]
                nSim[iz,iy,ix] =   nSim[iz,iy,ix]   +   1.
        mask    =   (nSim>=0.1)
        g1Sim[mask] =   g1Sim[mask]/nSim[mask]
        g2Sim[mask] =   g2Sim[mask]/nSim[mask]
        return g1Sim,g2Sim,nSim

    def process(self,cache,isim):
        self.log.info('processing simulation: %d' %isim)
        parser      =   cache.parser
        fieldName   =   parser.get('file','fieldN')
        simSrcName  =   './s16aPre/%s_RG_mock.fits' %(fieldName)
        simSrc      =   pyfits.getdata(simSrcName)
        #source z axis
        if parser.has_option('sourceZ','zname'):
            zname   =   parser.get('sourceZ','zname')
        else:
            zname   =   'z'
        zname   =   zname+'_%d' %isim
        g1name  =   'g1_%d'     %isim
        g2name  =   'g2_%d'     %isim
        g1Sim,g2Sim,nSim=   self.pixelize(simSrc,zname,g1name,g2name,parser)
        shearSim=   g1Sim+np.complex128(1j)*g2Sim
        return shearSim
    

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_argument('--configDir', type= str, 
                        default='./s16a3D/pix-0.05/nframe-3/lambda-3.5/',
                        help='directory for configuration files')
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
