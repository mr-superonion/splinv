#!/usr/bin/env python

import numpy as np
import astropy.io.fits as pyfits

from configparser import ConfigParser
from pixel3D import cartesianGrid3D

zer         =   'pz'        #pz: photo-z/zt: spect-z
ver         =   'nl20'      #number of lens plane
outfname    =   'planck-cosmo/lensing_kernel-%s-%s.fits' %(zer,ver)
datDir      =   '/work/xiangchong.li/work/S16ACatalogs/'
tn          =   '9347.fits' #tract directory

configName  =   'planck-cosmo/config-pix96-%s.ini' %ver

def main():
    parser      =   ConfigParser()
    parser.read(configName)
    gridInfo    =   cartesianGrid3D(parser)
    if zer=='pz':
        # Load necessary data from a HSC tract
        cfn     =   '%s/S16AStandardCalibrated/tract/%s' %(datDir,tn)
        cdata   =   pyfits.getdata(cfn)
        poz_best=   cdata['mlz_photoz_best']
        pfn     =   '%s/S16AStandardCalibrated/tract/%s_pofz.fits' %(datDir,tn.split('.')[0])
        poz_data=   pyfits.getdata(pfn)['PDF']
        bfn     =   '%s/S16AStandardV2/field/pz_pdf_bins_mlz.fits' %datDir
        poz_bins=   pyfits.getdata(bfn)['BINS']
        lensKernel= gridInfo.lensing_kernel(poz_grids=poz_bins,poz_data=poz_data,poz_best=poz_best)
    elif zer=='zt':
        lensKernel= gridInfo.lensing_kernel()
    else:
        print('we do not support the lensing kernel: %s' %zer)
    pyfits.writeto(outfname,lensKernel,overwrite=True)
    return

if __name__ == "__main__":
    main()
