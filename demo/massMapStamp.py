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
from astropy.table import Table,vstack
from sparseBase import massmapSparsityTask
from sparseBase import massmapSparsityTaskNew
from configparser import ConfigParser
from pixel3D import cartesianGrid3D

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

class massMapStampBatchConfig(pexConfig.Config):
    haloName=   pexConfig.Field(dtype=str,
                default='planck-cosmo/nfw-halos/haloCat-202010032144.csv',
                doc = 'halo catalog file name')
    configName  =   pexConfig.Field(dtype=str,
                default='planck-cosmo/config-pix96-nl20.ini',
                doc = 'configuration file name')
    outDir  =   pexConfig.Field(dtype=str,
                default='planck-cosmo/sparse-f3-3/',
                doc = 'output directory')
    pixDir  =   pexConfig.Field(dtype=str,
                default='planck-cosmo/pix96-ns10/',
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
        ss  =   pyascii.read(self.config.haloName)[Id]
        iz  =   ss['iz']
        im  =   ss['im']
        outDirH =   os.path.join(self.config.outDir,'halo%d%d'%(iz,im))
        if not os.path.isdir(outDirH):
            os.mkdir(outDirH)
        pool.storeSet(outDirH=outDirH)
        pool.storeSet(ss=ss)
        resList =   pool.map(self.process,range(100))
        resList =   [x for x in resList if x is not None]

        # peak catalogs list
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
        names   =   ('iz','iy','ix','value')
        c1,v1   =   detect3D.local_maxima_3D(delta)
        c2,v2   =   detect3D.local_minima_3D(delta)
        data1   =   np.hstack([c1,v1[:,None]])
        data2   =   np.hstack([c2,v2[:,None]])
        src     =   Table(data=np.vstack([data1,data2]),names=names)
        return src

    def map_reconstruct(self,cache,parser,isim):
        """
        Reconstruct density map
        """
        ss  =   cache.ss
        iz  =   ss['iz']
        im  =   ss['im']
        #pnm =   '%d%d-sim%d'%(iz,im,isim)
        pnm =   'sim%d' %(isim)
        outfname1=os.path.join(cache.outDirH,'deltaR-%s-lasso.fits' %pnm)
        outfname2=os.path.join(cache.outDirH,'deltaR-%s-alasso2.fits' %pnm)
        outfname3=os.path.join(cache.outDirH,'alphaR-%s-lasso.fits' %pnm)
        outfname4=os.path.join(cache.outDirH,'alphaR-%s-alasso2.fits' %pnm)
        if not (os.path.isfile(outfname1) and os.path.isfile(outfname1)):
            sparse3D    =   massmapSparsityTaskNew(parser)
            sparse3D.process(1500)
            sparse3D.reconstruct()
            delta1  =   sparse3D.deltaR
            pyfits.writeto(outfname1,delta1)
            pyfits.writeto(outfname3,sparse3D.alphaR)

            w   =   sparse3D.adaptive_lasso_weight(gamma=2.,sm_scale=0.)
            sparse3D.fista_gradient_descent(800,w=w)
            sparse3D.reconstruct()
            delta2  =   sparse3D.deltaR
            pyfits.writeto(outfname2,delta2)
            pyfits.writeto(outfname4,sparse3D.alphaR)
        else:
            self.log.info('Already have reconstructed map')
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

        pnm =   '%d%d-sim%d'%(iz,im,isim)
        #pnm =   'sim%d' %(isim)

        haloDir     =   'halo%d%d' %(iz,im)
        g1fname     =   os.path.join(cache.pixDir,haloDir,'pixShearR-g1-%s.fits' %pnm)
        g2fname     =   os.path.join(cache.pixDir,haloDir,'pixShearR-g2-%s.fits' %pnm)
        sigmafname  =   os.path.join(cache.pixDir,'pixStd.fits')
        lkfname     =   'planck-cosmo/lensing_kernel-pz-nl20.fits'

        parser.set('prepare','g1fname',g1fname)
        parser.set('prepare','g2fname',g2fname)
        parser.set('prepare','sigmafname',sigmafname)
        parser.set('prepare','lkfname',lkfname)

        parser.set('lensZ','resolve_lim','0.4') #pix
        parser.set('lensZ','rs_base','0.12')    #Mpc/h

        # Reconstruction Init
        lbd =   5.
        tau =   0.
        parser.set('sparse','lbd','%s' %lbd )
        parser.set('sparse','aprox_method','fista' )
        parser.set('sparse','nframe','3' )
        parser.set('sparse','minframe','0' )
        parser.set('sparse','tau','%s' %tau)
        parser.set('sparse','debugList','[]')
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
