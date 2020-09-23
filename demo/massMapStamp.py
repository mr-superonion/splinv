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
import detect3D
import numpy as np
import astropy.io.fits as pyfits
import astropy.io.ascii as pyascii
from pixel3D import cartesianGrid3D
from astropy.table import Table,vstack
from sparseBase import massmapSparsityTask
from configparser import ConfigParser

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

class massMapStampBatchConfig(pexConfig.Config):
    haloName=   pexConfig.Field(dtype=str, default='sims/haloCat-202009201646.csv',
                doc = 'halo catalog file name')
    configName  =   pexConfig.Field(dtype=str, default='config-pix96-nl15.ini',
                doc = 'configuration file name')
    outDir  =   pexConfig.Field(dtype=str, default='sparse-f1/',
                doc = 'output directory')
    pixDir  =   pexConfig.Field(dtype=str, default='pix96-nl15/',
                doc = 'output directory')

    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    def validate(self):
        pexConfig.Config.validate(self)
        assert os.path.isfile(self.haloName),\
            'Cannot find halo catalog file: %s' %self.haloName
        assert os.path.isfile(self.configName),\
            'Cannot find configure file: %s' %self.configName
        assert os.path.isdir(self.pixDir),\
            'Cannot find pixelaztion directory: %s' %self.pixDir

class massMapStampRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        # number of halos
        return [(ref, kwargs) for ref in range(64)]

def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)

class massMapStampBatchTask(BatchPoolTask):
    ConfigClass = massMapStampBatchConfig
    RunnerClass = massMapStampRunner
    _DefaultName = "massMapStampBatch"
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
        pool=   Pool("massMapStamp")
        pool.cacheClear()
        pool.storeSet(haloName=self.config.haloName)
        pool.storeSet(configName=self.config.configName)
        pool.storeSet(pixDir=self.config.pixDir)
        pool.storeSet(outDir=self.config.outDir)
        ss  =   pyascii.read(self.config.haloName)[Id]
        iz  =   ss['iz']
        im  =   ss['im']
        pool.storeSet(ss=ss)
        resList =   pool.map(self.process,range(100))
        resList =   [x for x in resList if x is not None]
        catA1   =   vstack([cat[0] for cat in resList])
        catA2   =   vstack([cat[1] for cat in resList])
        ofname1 =   os.path.join(self.config.outDir,'src-%d%d-lasso.fits' %(iz,im))
        ofname2 =   os.path.join(self.config.outDir,'src-%d%d-alasso2.fits' %(iz,im))
        catA1.write(ofname1,overwrite=True)
        catA2.write(ofname2,overwrite=True)
        return

    def process(self,cache,isim):
        """
        simulate shear field of halo and pixelize on postage-stamp
        @param cache    cache of the pool
        @param ss       halo source
        """

        parser  =   self.prepare(cache,isim)
        return self.map_reconstruct(cache,parser,isim)

    def source_detect(self,delta):
        """
        Detect halo
        """
        names=('iz','iy','ix','value')
        c,v =   detect3D.local_maxima_3D(delta)
        src=    Table(data=np.hstack([c,v[:,None]]),names=names)
        return src

    def map_reconstruct(self,cache,parser,isim):
        """
        Reconstruct density map
        """
        ss  =   cache.ss
        iz  =   ss['iz']
        im  =   ss['im']
        outfname1=os.path.join(cache.outDir,'deltaR-%d%d-sim%d-lasso.fits' %(iz,im,isim))
        outfname2=os.path.join(cache.outDir,'deltaR-%d%d-sim%d-alasso2.fits' %(iz,im,isim))
        if not (os.path.isfile(outfname1) and os.path.isfile(outfname1)):
            self.log.info('Already have reconstructed map')
            sparse3D    =   massmapSparsityTask(parser)
            sparse3D.process(1500)
            sparse3D.reconstruct()
            delta1  =   sparse3D.deltaR
            pyfits.writeto(outfname1,delta1)

            w=sparse3D.adaptive_lasso_weight(gamma=2.)
            sparse3D.fista_gradient_descent(800,w=w)
            sparse3D.reconstruct()
            delta2  =   sparse3D.deltaR
            pyfits.writeto(outfname2,delta2)
        else:
            delta1  =   pyfits.getdata(outfname1)
            delta2  =   pyfits.getdata(outfname2)

        src1    =   self.source_detect(delta1)
        src2    =   self.source_detect(delta2)
        src1['isim']=np.ones(len(src1),dtype=int)*isim
        src2['isim']=np.ones(len(src2),dtype=int)*isim
        gridInfo=   cartesianGrid3D(parser)
        self.physical_coords(src1,gridInfo,ss)
        self.physical_coords(src2,gridInfo,ss)
        return src1,src2

    def physical_coords(self,src,gridInfo,ss):
        dx  =   gridInfo.xbound[1]-gridInfo.xbound[0]
        dy  =   gridInfo.ybound[1]-gridInfo.ybound[0]
        dz  =   gridInfo.zlbound[1]-gridInfo.zlbound[0]
        x0  =   gridInfo.xbound[0]
        y0  =   gridInfo.ybound[0]
        z0  =   gridInfo.zlbound[0]

        src['dRA']   =   (x0+src['ix']*dx)*60.
        src['dDEC']  =   (y0+src['iy']*dy)*60.
        src['dZ']    =   z0+src['iz']*dz-ss['zh']
        return

    def prepare(self,cache,isim):
        """
        Prepare for the computation
        """
        ss  =   cache.ss
        iz  =   ss['iz']
        im  =   ss['im']
        self.log.info('processing halo iz: %d, im: %d, realization: %d' %(iz,im,isim))

        # pixelaztion class
        parser  =   ConfigParser()
        parser.read(cache.configName)

        g1fname     =   os.path.join(cache.pixDir,'pixShearR-g1-%d%d-sim%d.fits' %(iz,im,isim))
        g2fname     =   os.path.join(cache.pixDir,'pixShearR-g2-%d%d-sim%d.fits' %(iz,im,isim))
        sigmafname  =   os.path.join(cache.pixDir,'pixStd.fits')
        lkfname     =   os.path.join(cache.pixDir,'lensing_kernel.fits' )

        parser.set('prepare','g1fname',g1fname)
        parser.set('prepare','g2fname',g2fname)
        parser.set('prepare','sigmafname',sigmafname)
        parser.set('prepare','lkfname',lkfname)

        parser.set('lensZ','zmin','0.05')
        parser.set('lensZ','zscale','0.05')
        parser.set('lensZ','nlp','15')

        # Reconstruction Init
        tau=0.
        parser.set('sparse','lbd','5.' )
        parser.set('sparse','aprox_method','fista' )
        parser.set('sparse','nframe','1' )
        parser.set('sparse','minframe','0' )
        parser.set('sparse','tau','%s' %tau)
        parser.set('sparse','debugList','[]')
        parser.set('file','outDir',cache.outDir)
        return parser

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
