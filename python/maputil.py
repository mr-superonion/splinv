# Copyright 20200227 Xiangchong Li.
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
import os
import warnings
import cosmology
import numpy as np

import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from scipy.stats import norm as statnorm


# Field keys used in the noise variance file for simulations
field_keys = {'W02'     : 'XMM',
              'W03'     : 'GAMA09H',
              'W04-12'  : 'WIDE12H',
              'W04-15'  : 'GAMA15H',
              'W05'     : 'VVDS',
              'W06'     : 'HECTOMAP'}

field_names = {'XMM'    : 'W02',
              'GAMA09H' : 'W03',
              'WIDE12H' : 'W04_WIDE12H',
              'GAMA15H' : 'W04_GAMA15H',
              'VVDS'    : 'W05',
              'HECTOMAP': 'W06'}

field_nx    = {'XMM'    :   2,
              'GAMA09H' :   5,
              'WIDE12H' :   10,
              'GAMA15H' :   4,
              'VVDS'    :   7,
              'HECTOMAP':   4}

field_ny    = {'XMM'    :   2,
              'GAMA09H' :   2,
              'WIDE12H' :   2,
              'GAMA15H' :   1,
              'VVDS'    :   2,
              'HECTOMAP':   1}

def rotCatalog(e1, e2, phi=None):
    cs      =   np.cos(phi)
    ss      =   np.sin(phi)
    e1_rot  =   e1 * cs + e2 * ss
    e2_rot  =   (-1.0) * e1 * ss + e2 * cs
    return e1_rot, e2_rot

def plotPozAve(titlename,pozBin,pozAve,oname):
    #plot the average poz
    plt.close()
    for i in range(0,10):
        plt.plot(pozBin,pozAve[i],label=r'zbin:%d' %(i+1))
    plt.xlim(-0.1,3)
    plt.ylim(0.,0.082)
    plt.legend()
    plt.xlabel('z',fontsize=15)
    plt.ylabel('PDF',fontsize=15)
    plt.title(titlename)
    plt.grid()
    plt.tight_layout()
    plt.savefig(oname)
    plt.close()
    return

def plotLensKernel(titlename,zcgrid,lensKernel1,lensKernel2,oname):
    #plot the lensing kernel
    plt.close()
    cmap=plt.get_cmap('tab20')
    nzl=lensKernel1.shape[1]
    for i in range(0,nzl,2):
        norm1=np.sqrt(np.sum(lensKernel1[:,i]**2.))
        norm2=np.sqrt(np.sum(lensKernel2[:,i]**2.))
        plt.plot(zcgrid,lensKernel1[:,i]/norm1,'-',c=cmap(i))
        plt.plot(zcgrid,lensKernel2[:,i]/norm2,'--',c=cmap(i))
    plt.xlabel(r'$z_s$',fontsize=15)
    plt.ylabel('lensing kernel',fontsize=15)
    plt.grid()
    plt.title(titlename)
    plt.tight_layout()
    plt.savefig(oname)
    plt.close()
    return

def plotHist(fieldname,array,pixels,oname):
    # Plot the histogram of observable
    # both on galaxy level and on pixel level
    plt.close()
    cmap=plt.get_cmap('tab20')

    plt.figure(figsize=(8,6))

    gbin=plt.hist(array,bins=100,density=True,range=(-1.2,1.2),histtype='step',label='galaxy',color=cmap(0))[1]
    gbinGal=(gbin-np.average(array))/np.std(array)
    a=statnorm.pdf(gbinGal)
    plt.plot(gbin,a/np.sum(a)/(gbin[1]-gbin[0]),color=cmap(0),ls='--')

    gbin=plt.hist(pixels,bins=100,density=True,range=(-1.2,1.2),histtype='step',label='pixel',color=cmap(2))[1]

    gbinPix=(gbin-np.average(pixels))/np.std(pixels)
    a=statnorm.pdf(gbinPix)
    plt.plot(gbin,a/np.sum(a)/(gbin[1]-gbin[0]),color=cmap(2),ls='--')
    plt.title('%s' %fieldname,fontsize=20)
    plt.xlabel(r'$\delta g_1$',fontsize=20)
    plt.ylabel(r'$P(\delta g_1)$',fontsize=20)
    plt.legend(fontsize=20)
    plt.yscale('log')
    plt.ylim(1e-4,15)
    plt.grid()
    plt.tight_layout()
    plt.savefig(oname)
    plt.close()
    return
