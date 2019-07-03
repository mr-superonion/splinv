#!/usr/bin/env python
import os
import glob
import fitsio
import numpy as np
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

outFname=  'mlz_photoz_pdf_stack.fits' 
plot_bins   =   fitsio.read('/work/xiangchong.li/work/S16AStandard/S16A_pz_pdf/mlz/target_wide_s16a_wide12h_9832.0.P.fits',ext=2)['BINS']
if not os.path.exists(outFname):
    inDir   =  '/work/xiangchong.li/work/S16AStandard/S16AStandardCalibrated/tract' 
    gNames  =   os.path.join(inDir,'*_pofz.fits')
    nameList=   glob.glob(gNames)
    pdfStack=   np.zeros(len(plot_bins))
    numStack=   0.
    nName   =   len(nameList)
    for ifile,fname in enumerate(nameList):
        print("finish %.2f %" %(ifile/nName*100))
        dataTract=  fitsio.read(fname)['PDF']
        outData =   np.sum(dataTract,axis=0)
        pdfStack=   pdfStack+outData
        numStack+=  len(dataTract)
    pdfStack=   pdfStack/numStack
    fitsio.write(outFname,pdfStack)
else:
    pdfStack=   fitsio.read(outFname)

fig = plt.figure()
plt.plot(plot_bins,pdfStack)
plt.xlabel('redshift',fontsize=15)
plt.ylabel('prob',fontsize=15)
plt.tight_layout()
plt.savefig('redshift_source.eps')
