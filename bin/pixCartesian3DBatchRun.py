#!/usr/bin/env python
# pixelize data into 3D cartesian coordinates
from multiprocessing import Pool
from configparser import ConfigParser

import numpy as np
import astropy.io.fits as pyfits
from pixel3D import cartesianGrid3D

raname='raR'
decname='decR'
zname='zbest'
g1name='g1n'
g2name='g2n'
ngroup=100 # groups in the input fits file

configName  =   'stampSim/HSC-like/process/config.ini'
parser      =   ConfigParser()
parser.read(configName)
gridInfo    =   cartesianGrid3D(parser)

def process(isim):
    im = isim//9
    iz = isim%9
    infname='stampSim/HSC-like/stampSim-HSC_like-TJ03-%d,%d-202004021856.fits' %(iz,im)
    outfname='stampSim/HSC-like/process/pixShear-%d-%d.fits' %(iz,im)

    noiTab=pyfits.getdata(infname)
    ng=len(noiTab)//ngroup

    pixDatAll=np.zeros((ngroup,)+gridInfo.shape,dtype=np.complex128)

    for ig in range(ngroup):
        noiU=noiTab[ig*ng:(ig+1)*ng]
        val=noiU[g1name]+noiU[g2name]*1j
        pixDatAll[ig]=gridInfo.pixelize_data(noiU[raname],noiU[decname],noiU[zname],val)

    pyfits.writeto(outfname,pixDatAll)
    return

if __name__=="__main__":
    nproc=12
    nsim=81
    with Pool(nproc) as p:
        p.map(process,range(nsim))
