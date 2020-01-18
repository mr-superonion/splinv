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
minZList=   np.array([0.2,0.4,0.6,0.8])-0.03
maxZList=   np.array([0.2,0.4,0.6,0.8])+0.03
nPlot   =   len(minZList)
outFname=  'mlz_photoz_pdf_stack4.fits' 
plot_bins   =   fitsio.read('/work/xiangchong.li/work/S16AStandard/S16A_pz_pdf/mlz/target_wide_s16a_wide12h_9832.0.P.fits',ext=2)['BINS']
if not os.path.exists(outFname):
    inDir   =  '/work/xiangchong.li/work/S16AStandard/S16AStandardCalibrated/tract' 
    gNames  =   os.path.join(inDir,'*_pofz.fits')
    nameList=   glob.glob(gNames)
    pdfStack=   np.zeros((nPlot,len(plot_bins)))
    numStack=   np.zeros(nPlot)
    nName   =   len(nameList)
    for ifile,fname in enumerate(nameList):
        print("finish %0.2f percent" %(ifile/nName*100.))
        dataTract=  fitsio.read(fname)['PDF']
        for ip in range(nPlot):
            zMax    =   plot_bins[np.argmax(dataTract,axis=1)]
            mask    =   (zMax>minZList[ip])&(zMax<maxZList[ip])
            dataMask=   dataTract[mask]
            outData =   np.sum(dataMask,axis=0)
            pdfStack[ip]+=   outData
            numStack[ip]+=  len(dataTract)
    pdfStack=   pdfStack/numStack[:,None]
    fitsio.write(outFname,pdfStack)
    fitsio.write('pozBins.fits',plot_bins)
else:
    pdfStack=   fitsio.read(outFname)

fig =   plt.figure()
ax  =   fig.add_subplot(2,2,1)
ax.set_title(r'z $\sim$ 0.2')
ax.plot(plot_bins,pdfStack[0])
ax.set_xlabel('redshift',fontsize=15)
ax.set_ylabel('prob',fontsize=15)
ax  =   fig.add_subplot(2,2,2)
ax.set_title(r'z $\sim$ 0.4')
ax.plot(plot_bins,pdfStack[1])
ax.set_xlabel('redshift',fontsize=15)
ax.set_ylabel('prob',fontsize=15)
ax  =   fig.add_subplot(2,2,3)
ax.set_title(r'z $\sim$ 0.6')
ax.plot(plot_bins,pdfStack[2])
ax.set_xlabel('redshift',fontsize=15)
ax.set_ylabel('prob',fontsize=15)
ax  =   fig.add_subplot(2,2,4)
ax.set_title(r'z $\sim$ 0.8')
ax.plot(plot_bins,pdfStack[3])
ax.set_xlabel('redshift',fontsize=15)
ax.set_ylabel('prob',fontsize=15)
plt.tight_layout()
plt.savefig('redshift_source4.eps')
