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
from astropy.table import Table
from configparser import ConfigParser

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

class haloSimStampConfig(pexConfig.Config):
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    
    def validate(self):
        pexConfig.Config.validate(self)

class haloSimStampTask(pipeBase.CmdLineTask):
    _DefaultName = "haloSimStamp"
    ConfigClass = haloSimStampConfig
    def __init__(self,**kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)

    @pipeBase.timeMethod
    def run(self,configName,simDir):
        configName  =   configName[:-1]
        self.log.info(configName)
        parser      =   ConfigParser()
        parser.read(os.path.join(simDir,configName))
        # cosmology
        omega_m     =   parser.getfloat('cosmology','omega_m')
        omega_l     =   parser.getfloat('cosmology','omega_l')
        h_cos       =   parser.getfloat('cosmology','h_cos')
        # sources
        size        =   parser.getfloat('sources','size')       #(arcmin)
        ns_per_arcmin=  parser.getint('sources','ns_per_arcmin')
        var_gErr    =   parser.getfloat('sources','var_gErr')
        # lens
        z_cl        =   parser.getfloat('lens','z_cl')                 #redshift
        x_cl        =   parser.getfloat('lens','x_cl')                 #arcsec
        y_cl        =   parser.getfloat('lens','y_cl')                 #arcsec
        M_200       =   parser.getfloat('lens','M_200')                #(M_sun/h)
        #  sources
        ns          =   int(size**2.*ns_per_arcmin+1)
        x_s         =   np.random.random(ns)*size-size/2.
        y_s         =   np.random.random(ns)*size-size/2.
        z_s         =   self.getS16aZ(ns)
        #  halo
        pos_cl      =   galsim.PositionD(x_cl,y_cl)
        conc        =   6.02*(M_200/1.E13)**(-0.12)*(1.47/(1.+z_cl))**(0.16)
        halo        =   galsim.nfw_halo.NFWHalo(mass= M_200,conc=conc,
                        redshift= z_cl,halo_pos=pos_cl,omega_m= omega_m,
                        omega_lam= omega_l)
        kappa_s     =   halo.getConvergence(pos=(x_s,y_s),
                        z_s=z_s,units = "arcmin")
        g1_s,g2_s   =   halo.getShear(pos=(x_s,y_s),
                        z_s=z_s,units="arcmin",reduced=False)
        if var_gErr >=1.e-5:
            np.random.seed(100)
            g1_noi  =   np.random.randn(ns)*var_gErr
            g2_noi  =   np.random.randn(ns)*var_gErr
            g1_s    =   g1_s+g1_noi
            g2_s    =   g2_s+g2_noi

        # write the data
        data        =   (x_s,y_s,z_s*np.ones(ns),g1_s,g2_s,kappa_s)
        sources     =   Table(data=data,names=('ra','dec','z','g1','g2','kappa'))
        outFname    =   'src_'+configName.split('.')[0]+'.fits'
        outFname    =   os.path.join(simDir,outFname)
        sources.write(outFname)
        return

    def getS16aZ(self,nobj):
        z_bins      =   fitsio.read('/work/xiangchong.li/work/S16AStandard/S16A_pz_pdf/mlz/target_wide_s16a_wide12h_9832.0.P.fits',ext=2)['BINS']
        pdf         =   fitsio.read('/work/xiangchong.li/work/massMapSim/mlz_photoz_pdf_stack.fits')
        pdf         =   pdf.astype(float)
        nbin        =   len(pdf)
        pdf         /=  np.sum(pdf)
        cdf         =   np.empty(nbin,dtype=float)
        np.cumsum(pdf,out=cdf)
                  
        # Monte Carlo z
        r       =   np.random.random(size=nobj)
        tzmc    =   np.empty(nobj, dtype=float)
        tzmc    =   np.interp(r, cdf, z_bins)
        return tzmc

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

class haloSimStampBatchConfig(pexConfig.Config):
    haloSimStamp = pexConfig.ConfigurableField(
        target = haloSimStampTask,
        doc = "haloSimStamp task to run on multiple cores"
    )
    simDir  =   pexConfig.Field(dtype=str, default='simulation9',
                doc = 'directory to store exposures')
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    
    def validate(self):
        pexConfig.Config.validate(self)
        if not os.path.exists(self.simDir):
            self.log.info('Do not have %s' %self.simDir)
    
class haloSimStampRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [(ref, kwargs) for ref in range(1)] 

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
        self.makeSubtask("haloSimStamp")
    
    @abortOnError
    def run(self,Id):
        #Prepare the pool
        simDir  =   self.config.simDir
        pool    =   Pool("haloSimStamp")
        pool.cacheClear()
        pool.storeSet(simDir=self.config.simDir)
        configList= os.popen('ls %s/ |grep ini' %simDir).readlines()
        pool.map(self.process,configList)
        self.log.info('finish group %d'%(Id))
        return
        
    def process(self,cache,ifield):
        self.haloSimStamp.run(ifield,cache.simDir)
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
