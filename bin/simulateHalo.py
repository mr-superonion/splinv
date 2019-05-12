#!/usr/bin/env python
import os
import fitsio
import galsim
import numpy as np
from astropy.table import Table


simDir  =   './simulation/'
if not os.path.exists(simDir):
    os.mkdir(simDir)

#For stamp
ngrid   =   64  
pix_scale   =   0.5#(arcmin/pix)
ns_per_arcmin=  40
ns          =   int((pix_scale*ngrid)**2.*ns_per_arcmin+1)

#For halo
omega_m =   0.3
omega_L =   0.7
h_cos   =   0.7
x_cl    =   0.
y_cl    =   0.
pos_cl  =   galsim.PositionD(x_cl,y_cl)*60
z_cl    =   0.05
M_200   =   1.e14*h_cos #(M_sun/h)
conc    =   6.02*(M_200/1.E13)**(-0.12)*(1.47/(1.+z_cl))**(0.16)
halo    =   galsim.nfw_halo.NFWHalo(mass= M_200,
            conc=conc, redshift= z_cl,
            halo_pos=pos_cl ,omega_m= omega_m,
            omega_lam= omega_L)
#For sources
x_s     =   (np.random.random(ns)*ngrid-ngrid/2.)*pix_scale+x_cl
y_s     =   (np.random.random(ns)*ngrid-ngrid/2.)*pix_scale+y_cl
z_s     =   0.8 
kappa_s =   halo.getConvergence(pos=(x_s,y_s),z_s=z_s,units = "arcmin")
g1_s,g2_s=  halo.getShear(pos=(x_s,y_s),z_s=z_s,units = "arcmin",reduced=False)
data    =   (x_s,y_s,z_s*np.ones(ns),g1_s,g2_s,kappa_s)
print(data)

sources =   Table(data=data,names=('ra','dec','z','g1','g2','kappa'))
sources.write(os.path.join(simDir,'src.fits'))


ngrid2  =   ngrid*2
g1Map_true   =   np.zeros((ngrid2,ngrid2))  
g2Map_true   =   np.zeros((ngrid2,ngrid2))  
kMap_true    =   np.zeros((ngrid2,ngrid2))  
numMap       =   np.zeros((ngrid2,ngrid2),dtype=np.int)  
maskMap      =   np.zeros((ngrid2,ngrid2),dtype=np.int)  
xMin    =   x_cl-ngrid*pix_scale
yMin    =   y_cl-ngrid*pix_scale

for iss in range(ns):
    ix  =   int((x_s[iss]-xMin)//pix_scale)
    iy  =   int((y_s[iss]-yMin)//pix_scale)
    g1Map_true[iy,ix]   =   g1Map_true[iy,ix]+g1_s[iss]
    g2Map_true[iy,ix]   =   g2Map_true[iy,ix]+g2_s[iss]
    kMap_true[iy,ix]    =   kMap_true[iy,ix]+kappa_s[iss]
    numMap[iy,ix]       =   numMap[iy,ix]+1.

for j in range(ngrid2):
    for i in range(ngrid2):
        if numMap[j,i]!=0:
            g1Map_true[j,i] =   g1Map_true[j,i]/numMap[j,i]
            g2Map_true[j,i] =   g2Map_true[j,i]/numMap[j,i]
            kMap_true[j,i]  =   kMap_true[j,i]/numMap[j,i]
            maskMap[j,i]    =   1

fitsio.write(os.path.join(simDir,'g1Map_true.fits'),g1Map_true)
fitsio.write(os.path.join(simDir,'g2Map_true.fits'),g2Map_true)
fitsio.write(os.path.join(simDir,'kMap_true.fits'),kMap_true)
fitsio.write(os.path.join(simDir,'maskMap.fits'),maskMap)
fitsio.write(os.path.join(simDir,'numMap.fits'),numMap)
