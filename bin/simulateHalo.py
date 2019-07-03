#!/usr/bin/env python
import os
import fitsio
import galsim
import numpy as np
from astropy.table import Table

def getS16aZ(nobj):
    z_bins  =   fitsio.read('/work/xiangchong.li/work/S16AStandard/S16A_pz_pdf/mlz/target_wide_s16a_wide12h_9832.0.P.fits',ext=2)['BINS']
    pdf     =   fitsio.read('/work/xiangchong.li/work/massMapSim/mlz_photoz_pdf_stack.fits')
    pdf     =   pdf.astype(float)
    nbin    =   len(pdf)
    pdf     /=  np.sum(pdf)
    cdf     =   np.empty(nbin,dtype=float)
    np.cumsum(pdf,out=cdf)
              
    # Monte Carlo z
    r       =   np.random.random(size=nobj)
    tzmc    =   np.empty(nobj, dtype=float)
    tzmc    =   np.interp(r, cdf, z_bins)
    tzmc[tzmc<0.06]=0.06
    return tzmc


nhalo   =   1
isim    =   5
simDir  =   './simulation%s/' %(isim)
if not os.path.exists(simDir):
    os.mkdir(simDir)

#For sources
size        =   32 #(arcmin)
ns_per_arcmin=  40
ns          =   int(size**2.*ns_per_arcmin+1)
var_gErr    =   0#0.25
x_s         =   np.random.random(ns)*size-size/2.
y_s         =   np.random.random(ns)*size-size/2.
z_s         =   getS16aZ(ns)#0.8 
kappa_s     =   np.zeros(ns)
g1_s        =   np.zeros(ns)
g2_s        =   np.zeros(ns)
#Second sources for high resolution true map
x_s2        =   np.random.random(ns*16)*size-size/2.
y_s2        =   np.random.random(ns*16)*size-size/2.
z_s2        =   getS16aZ(ns*16)
kappa_s2    =   np.zeros(ns*16)

#For halo
halos   =   []
omega_m =   0.3
omega_L =   0.7
h_cos   =   0.7
z_cl    =   np.array([0.05,0.12,0.1]) #redshift
x_cl    =   np.array([0.,5.,-3.])*60. #arcsec
y_cl    =   np.array([0.,8.,2.])*60.  #arcsec
M_200   =   np.array([1.e14,1.8e13,7.e12])*h_cos #(M_sun/h)
for i in range(nhalo):#we use three halos
    pos_cl  =   galsim.PositionD(x_cl[i],y_cl[i])
    conc    =   6.02*(M_200[i]/1.E13)**(-0.12)*(1.47/(1.+z_cl[i]))**(0.16)
    halo    =   galsim.nfw_halo.NFWHalo(mass= M_200[i],
            conc=conc, redshift= z_cl[i],
            halo_pos=pos_cl ,omega_m= omega_m,
            omega_lam= omega_L)
    kappa_s_0   =   halo.getConvergence(pos=(x_s,y_s),z_s=z_s,units = "arcmin")
    g1_s_0,g2_s_0=  halo.getShear(pos=(x_s,y_s),z_s=z_s,units = "arcmin",reduced=False)
    kappa_s2_0  =   halo.getConvergence(pos=(x_s2,y_s2),z_s=z_s2,units = "arcmin")
    kappa_s =   kappa_s+ kappa_s_0
    g1_s    =   g1_s+ g1_s_0
    g2_s    =   g2_s+ g2_s_0
    kappa_s2=   kappa_s2+  kappa_s2_0


if var_gErr >=1.e-5:
    np.random.seed(100)
    g1_noi  =   np.random.randn(ns)*var_gErr
    g2_noi  =   np.random.randn(ns)*var_gErr
    g1_s    =   g1_s+g1_noi
    g2_s    =   g2_s+g2_noi

# write the data
data        =   (x_s,y_s,z_s*np.ones(ns),g1_s,g2_s,kappa_s)
sources     =   Table(data=data,names=('ra','dec','z','g1','g2','kappa'))
sources.write(os.path.join(simDir,'src.fits'))


#For stamp
ngList      =   [64,128,256]
for ngrid in ngList: #(pix)
    pix_scale   =   size/ngrid#(arcmin/pix)
    ngrid2      =   ngrid*2
    g1Map_true  =   np.zeros((ngrid2,ngrid2))
    g2Map_true  =   np.zeros((ngrid2,ngrid2))
    kMap_true   =   np.zeros((ngrid2,ngrid2))
    kMap_true2  =   np.zeros((ngrid2,ngrid2))# this is the true kMap
    numMap      =   np.zeros((ngrid2,ngrid2),dtype=np.int)  
    numMap2     =   np.zeros((ngrid2,ngrid2),dtype=np.int)  
    xMin        =   -size
    yMin        =   -size

    for iss in range(ns):
        ix  =   int((x_s[iss]-xMin)//pix_scale)
        iy  =   int((y_s[iss]-yMin)//pix_scale)
        g1Map_true[iy,ix]   =   g1Map_true[iy,ix]+g1_s[iss]
        g2Map_true[iy,ix]   =   g2Map_true[iy,ix]+g2_s[iss]
        kMap_true[iy,ix]    =   kMap_true[iy,ix]+kappa_s[iss]
        numMap[iy,ix]       =   numMap[iy,ix]+1.

    for iss in range(ns*16):
        ix  =   int((x_s2[iss]-xMin)//pix_scale)
        iy  =   int((y_s2[iss]-yMin)//pix_scale)
        kMap_true2[iy,ix]   =   kMap_true2[iy,ix]+kappa_s2[iss]
        numMap2[iy,ix]      =   numMap2[iy,ix]+1.
    for j in range(ngrid2):
        for i in range(ngrid2):
            if numMap[j,i]!=0:
                g1Map_true[j,i] =   g1Map_true[j,i]
                g2Map_true[j,i] =   g2Map_true[j,i]
                #g1 and g2 do not divide numMap
                kMap_true[j,i]  =   kMap_true[j,i]/numMap[j,i]
            if numMap2[j,i]!=0:
                kMap_true2[j,i]  =   kMap_true2[j,i]/numMap2[j,i]


    fitsio.write(os.path.join(simDir,'g1Map_grid%s.fits' %(ngrid)),g1Map_true)
    fitsio.write(os.path.join(simDir,'g2Map_grid%s.fits' %(ngrid)),g2Map_true)
    fitsio.write(os.path.join(simDir,'kMap_grid%s.fits' %(ngrid)),kMap_true)
    fitsio.write(os.path.join(simDir,'numMap_grid%s.fits' %(ngrid)),numMap)
    fitsio.write(os.path.join(simDir,'kMap_true%s.fits' %(ngrid)),kMap_true2)
