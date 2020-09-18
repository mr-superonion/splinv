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
import glob
import fitsio
import halSim
import numpy as np
from astropy.table import Table
from configparser import ConfigParser

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

class haloSimOneBatchConfig(pexConfig.Config):
    simDir  =   pexConfig.Field(dtype=str, default='oneHaloSim',
                doc = 'directory to store exposures')
    doDemo  =   pexConfig.Field(dtype=bool, default=True,
                doc = 'whether to do demostration')
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    def validate(self):
        pexConfig.Config.validate(self)
        assert os.path.exists(self.simDir),\
            'do not have simulation root directory: %s' %self.simDir

class haloSimOneRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [(ref, kwargs) for ref in range(1)] 
def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)
class haloSimOneBatchTask(BatchPoolTask):
    ConfigClass = haloSimOneBatchConfig
    RunnerClass = haloSimOneRunner
    _DefaultName = "haloSimOneBatch"
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
        """
        #Prepare the pool
        """
        simDir  =   self.config.simDir
        pool    =   Pool("haloSimOne")
        pool.cacheClear()
        pool.storeSet(simDir=self.config.simDir)
        configList= glob.glob('%s/*.ini' %simDir)
        self.log.info('number of data: %d' %len(configList))
        pool.map(self.process,configList)
        return

    def process(self,cache,configName):
        """
        @param cache    cache of the pool
        @param cache    relative directory of configure file
        """
        simDir      =   cache.simDir
        parser      =   ConfigParser()
        self.log.info('The configuration file used is: %s' %configName)
        parser.read(configName)

        """
        # Cosmology
        """
        omega_m     =   parser.getfloat('cosmology','omega_m')
        omega_l     =   parser.getfloat('cosmology','omega_l')
        h_cos       =   parser.getfloat('cosmology','h_cos')

        """
        # Sources
        """
        size        =   parser.getfloat('sources','size')       #(arcmin)
        ns_per_arcmin=  parser.getint('sources','ns_per_arcmin')
        ns          =   int(size**2.*ns_per_arcmin+1)
        self.log.info('number of galaxies: %d' %ns)
        x_s         =   np.random.random(ns)*size-size/2.*60    #(arcsec)
        y_s         =   np.random.random(ns)*size-size/2.*60    #(arcsec)
        z_s         =   self.getS16aZ(ns)
        var_gErr    =   parser.getfloat('sources','var_gErr')

        """
        # Lens Halo
        """
        z_cl        =   parser.getfloat('lens','z_cl')                 #redshift
        x_cl        =   parser.getfloat('lens','x_cl')*60.             #arcsec
        y_cl        =   parser.getfloat('lens','y_cl')*60.             #arcsec
        M_200       =   parser.getfloat('lens','M_200')                #(M_sun/h)
        conc        =   6.02*(M_200/1.E13)**(-0.12)*(1.47/(1.+z_cl))**(0.16)# 
        #halo        =   halSim.nfw_lensWB00(mass=M_200,conc=conc,redshift=z_cl,ra=x_cl,dec=y_cl)#omega_m=0.3
        halo        =   halSim.nfw_lensTJ03(mass=M_200,conc=conc,redshift=z_cl,ra=x_cl,dec=y_cl)#omega_m=0.3

        """
        # Prepare for Demo
        """
        if self.config.doDemo:
            """
            # For for transverse Plane
            """
            ngrid   =   parser.getint('transPlane','nx')
            pix_scale=  parser.getfloat('transPlane','scale')*60.   #[arcsec]
            xcG     =   parser.getfloat('transPlane','xcG')*60.   #[arcsec]
            ycG     =   parser.getfloat('transPlane','ycG')*60.   #[arcsec]
            """
            # For lens z axis
            """
            nlp     =   parser.getint('lensZ','nlp')
            zlMin   =   parser.getfloat('lensZ','zlmin')
            zlscale =   parser.getfloat('lensZ','zlscale')
            zBin0   =   np.arange(zlMin,zlMin+zlscale*nlp,zlscale)
            zBin1   =   zBin0+zlscale
            zhB     =   np.where((z_cl>zBin0)&(z_cl<zBin1))
            SigmaM  =   halo.Sigma_M_bin(zBin0,zBin1)
            """
            # Initialize the 3D demos
            """
            delta3D =   np.zeros((nlp,ngrid,ngrid))
            dDelta3D=   np.zeros((nlp,ngrid,ngrid))

        """
        # Simulate galaxy catalog
        """
        lensKer     =   halo.lensKernel(z_s)
        assert len(lensKer)==len(x_s),\
            'redshift and ra,dec does not have same length'
        kappa_s     =   lensKer*halo.Sigma(x_s,y_s)
        gamma       =   lensKer*halo.DeltaSigmaComplex(x_s,y_s)
        g1_s        =   gamma.real
        g2_s        =   gamma.imag
        if var_gErr >=1.e-5:
            np.random.seed(100)
            g1_noi  =   np.random.randn(ns)*var_gErr
            g2_noi  =   np.random.randn(ns)*var_gErr
            g1_s    =   g1_s+g1_noi
            g2_s    =   g2_s+g2_noi

        if self.config.doDemo:
            """
            # Record 3D delta Field
            """
            delta3D[zhB,:,:]+=  halo.SigmaAtom(pix_scale,ngrid,xcG,ycG)/SigmaM[zhB,None,None]
            a = delta3D[delta3D>1.e-10]
            """
            # Record 3D Dletadelta Field
            """
            dDelta3D[zhB,:,:]+= halo.DeltaSigmaAtom(pix_scale,ngrid,xcG,ycG)/SigmaM[zhB,None,None]

        """
        # write the data
        # catalog
        """
        data        =   (x_s,y_s,z_s*np.ones(ns),g1_s,g2_s,kappa_s)
        sources     =   Table(data=data,names=('ra','dec','z','g1','g2','kappa'))
        outFname    =   'src.fits'
        outFname    =   os.path.join(simDir,outFname)
        sources.write(outFname,overwrite=True)
        """
        # Domo
        """
        if self.config.doDemo:
            outD3name=  os.path.join(simDir,'delta3D.fits')
            fitsio.write(outD3name,delta3D,clobber=True)
            outdD3name=  os.path.join(simDir,'dDelta3D.fits')
            fitsio.write(outdD3name,dDelta3D,clobber=True)
        return

    def getS16aZ(self,nobj):
        """
        # Load the stacked poz
        @param mass     number of galaxies
        """
        z_bins      =   fitsio.read('/data2b/work/xiangchong.li/S16AStandard/S16A_pz_pdf/mlz/target_wide_s16a_wide12h_9832.0.P.fits',ext=2)['BINS']
        pdf         =   fitsio.read('/lustre2/work/xiangchong.li/massMapSim/mlz_photoz_pdf_stack.fits')
        pdf         =   pdf.astype(float)
        nbin        =   len(pdf)
        pdf         /=  np.sum(pdf)
        cdf         =   np.empty(nbin,dtype=float)
        np.cumsum(pdf,out=cdf)
        """
        # Monte Carlo z
        """
        r       =   np.random.random(size=nobj)
        tzmc    =   np.empty(nobj, dtype=float)
        tzmc    =   np.interp(r, cdf, z_bins)
        return tzmc

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
