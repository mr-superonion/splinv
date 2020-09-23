#!/usr/bin/env python
# pixelize data into 3D cartesian coordinates
import gc
from multiprocessing import Pool
from configparser import ConfigParser

import numpy as np
import astropy.io.fits as pyfits
from pixel3D import cartesianGrid3D

raname  =   'raR'
decname =   'decR'
zname   =   'zbest'
g1name  =   'g1R'
g2name  =   'g2R'
ngroup  =   100  # groups in the input fits file

configName  =   'stampSim/HSC-like/process-equalNum10/config.ini'
parser      =   ConfigParser()
parser.read(configName)
gridInfo    =   cartesianGrid3D(parser)

def pixelize_shear(isim):
    im  =   isim//9
    iz  =   isim%9
    outfname1   =  'stampSim/HSC-like/process-equalNum/pixShearR-g1-%d-%d.fits' %(iz,im)
    outfname2   =  'stampSim/HSC-like/process-equalNum/pixShearR-g2-%d-%d.fits' %(iz,im)
    if os.path.isfile(outfname1):
        print('Already have output for simulation: %d' %isim)
        return

    infname =   'stampSim/HSC-like/sims/stampSim-HSC_like-TJ03-%d,%d-202004021856.fits' %(iz,im)
    datTab  =   pyfits.getdata(infname)
    ng=len(datTab)//ngroup

    pixDatAll=np.zeros((ngroup,)+gridInfo.shape,dtype=np.complex128)
    for ig in range(ngroup):
        datU=datTab[ig*ng:(ig+1)*ng]
        val=(datU[g1name]+datU['g1n'])+(datU[g2name]+datU['g2n'])*1j
        outcome=gridInfo.pixelize_data(datU[raname],datU[decname],datU[zname],val)
        pixDatAll[ig]=outcome

    pyfits.writeto(outfname1,pixDatAll.real,overwrite=True)
    pyfits.writeto(outfname2,pixDatAll.imag,overwrite=True)
    del pixDatAll
    del datTab
    gc.collect()
    return

def estimate_sigma():
    infname =   'stampSim/HSC-like/process-equalNum/pixShearR-g1-0-0.fits'
    data    =   pyfits.getdata(infname)
    std     =   np.std(data,axis=0)*np.sqrt(2.)
    pyfits.writeto('stampSim/HSC-like/process-equalNum/pixStd.fits')
    return

if __name__=="__main__":
    nproc=12
    nsim=81
    with Pool(nproc) as p:
        p.map(pixelize_shear,range(nsim))
    estimate_sigma()
