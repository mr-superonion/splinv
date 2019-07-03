#!/usr/bin/env python
import os
import fitsio
import astropy.table as astTab
import numpy as np
from multiprocessing import Pool

pix_scale   =   1./60
z_scale     =   0.1
zMin        =   0.1
zMax        =   3.1
nz          =   20

def prepareHSCField16(fieldname):
    if fieldname    ==  'AEGIS':
        return
    inameRG =   './s16aPre2D/%s_RG.fits' %(fieldname)
    dataIn  =   fitsio.read(inameRG)
    xMax    =   np.max(dataIn['ra'])+pix_scale/2.
    xMin    =   np.min(dataIn['ra'])-pix_scale/2.
    yMax    =   np.max(dataIn['dec'])+pix_scale/2.
    yMin    =   np.min(dataIn['dec'])-pix_scale/2.
    nx      =   int((xMax-xMin)/pix_scale+0.5)
    ny      =   int((yMax-yMin)/pix_scale+0.5)
    print('%s: nx: %d, ny: %d, nz: %d' %(fieldname,nx,ny,nz))
    numGrid =   np.zeros((nz,ny,nx))
    g1Grid  =   np.zeros((nz,ny,nx))
    g2Grid  =   np.zeros((nz,ny,nx))
    xGrid   =   np.zeros((nz,ny,nx))
    yGrid   =   np.zeros((nz,ny,nx))
    zGrid   =   np.zeros((nz,ny,nx))
    zVGrid  =   np.zeros((nz,ny,nx))
    for ss in dataIn:
        ix  =   int((ss['ra']-xMin)/pix_scale)
        iy  =   int((ss['dec']-yMin)/pix_scale)
        iz  =   int((ss['mlz_photoz_best']-zMin)/z_scale)
        if iz>=0 and iz<nz:
            g1Grid[iz,iy,ix]=g1Grid[iz,iy,ix]+ss['g1']
            g2Grid[iz,iy,ix]=g2Grid[iz,iy,ix]+ss['g2']
            xGrid[iz,iy,ix]=xGrid[iz,iy,ix]+ss['ra']
            yGrid[iz,iy,ix]=yGrid[iz,iy,ix]+ss['dec']
            zGrid[iz,iy,ix]=zGrid[iz,iy,ix]+ss['mlz_photoz_best']
            zVGrid[iz,iy,ix]=zVGrid[iz,iy,ix]+ss['mlz_photoz_std_best']**2.
            numGrid[iz,iy,ix]=numGrid[iz,iy,ix]+1.
    mask    =   numGrid>0.1
    g1Grid  =   g1Grid[mask]
    g2Grid  =   g2Grid[mask]
    xGrid   =   xGrid[mask]
    yGrid   =   yGrid[mask]
    zGrid   =   zGrid[mask]
    zVGrid  =   zVGrid[mask]
    numGrid =   numGrid[mask]
    g1Grid  =   g1Grid/numGrid
    g2Grid  =   g2Grid/numGrid
    xGrid   =   xGrid/numGrid
    yGrid   =   yGrid/numGrid
    zGrid   =   zGrid/numGrid
    zVGrid  =   np.sqrt(zVGrid)/numGrid
    dataTab =   [xGrid,yGrid,zGrid,zVGrid,g1Grid,g2Grid] 
    names   =   ['ra','dec','mlz_photoz_best','mlz_photoz_std_best','g1','g2']
    outTab  =   astTab.Table(data=dataTab,names=names)
    onameRG =   './s16aPre3D/%s_RG.fits' %(fieldname)
    outTab.write(onameRG)
    return

def main():
    pool    =   Pool(1)
    fields  =   np.load('/work/xiangchong.li/work/S16AFPFS/dr1FieldTract.npy').item().keys() 
    pool.map(prepareHSCField16,fields)
    return

if __name__ == '__main__':
    main()
