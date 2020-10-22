#!/usr/bin/env python

import os
import gc
import numpy as np
import astropy.table as astTab
import astropy.io.fits as pyfits

datDir  =   "/work/xiangchong.li/work/S16ACatalogs/"
tn      =   '9347.fits'

def rotCatalog(e1, e2, phi=None):
    if phi  ==  None:
        phi = 2.0 * np.pi * np.random.rand(len(e1))
    cs = np.cos(phi)
    ss = np.sin(phi)
    e1_rot = e1 * cs + e2 * ss
    e2_rot = (-1.0) * e1 * ss + e2 * cs
    return e1_rot, e2_rot

def measShear(cdata):
    e1      =   cdata['ishape_hsm_regauss_e1']
    e2      =   cdata['ishape_hsm_regauss_e2']
    m_b     =   cdata['ishape_hsm_regauss_derived_shear_bias_m']
    c1_b    =   cdata['ishape_hsm_regauss_derived_shear_bias_c1']
    c2_b    =   cdata['ishape_hsm_regauss_derived_shear_bias_c2']
    weight  =   cdata['ishape_hsm_regauss_derived_shape_weight']
    w_A     =   np.sum(weight)
    m_bA    =   np.sum(m_b*weight)/w_A
    Res_A   =   1.-np.sum(cdata['ishape_hsm_regauss_derived_rms_e']**2.*weight)/w_A
    g1      =   1./(1.+m_bA)*(e1/2./Res_A-c1_b)
    g2      =   1./(1.+m_bA)*(e2/2./Res_A-c2_b)
    #t       =   1./(1.+m_bA)/2./Res_A
    #weight  =   weight/(t**2.)
    return g1,g2


def main():
    # read the data from S16A
    pfn =   os.path.join(datDir,'S16AStandardCalibrated/tract/%s_pofz.fits' %tn.split('.')[0])
    cfn =   os.path.join(datDir,'S16AStandardCalibrated/tract/%s' %tn)
    cdata=  pyfits.getdata(cfn)
    pdata=  pyfits.getdata(pfn)['PDF']
    assert len(pdata)==len(cdata)

    # mask the catalog data
    ra      =   cdata['coord_ra']*180/np.pi-np.average(cdata['coord_ra'])*180/np.pi
    dec     =   cdata['coord_dec']*180/np.pi-np.average(cdata['coord_dec'])*180/np.pi
    mask    =   (np.absolute(ra)<0.5) & (np.absolute(dec)<0.5)
    cdata   =   cdata[mask]
    pdata   =   pdata[mask]
    ra      =   ra[mask]
    dec     =   dec[mask]
    raR     =   np.random.rand(len(ra))-0.5
    decR    =   np.random.rand(len(dec))-0.5
    zBest   =   cdata['mlz_photoz_best']
    g1,g2   =   measShear(cdata)

    # bins boundarys for poz
    bfn=os.path.join(datDir,'S16AStandardV2/field/pz_pdf_bins_mlz.fits')
    poz_bins=pyfits.getdata(bfn)['BINS']
    nobj, nbin = pdata.shape
    assert len(poz_bins)==nbin


    pdata   =   pdata.astype(float)
    pdata   /=  np.sum(pdata,axis=1).reshape(nobj, 1)
    cdf     =   np.empty(shape=(nobj, nbin), dtype=float)
    np.cumsum(pdata, axis=1, out=cdf)

    mRange  =   range(100,1000)
    names   =   ('raH','decH','raR','decR','g1n','g2n','zbest','ztrue')

    for imock in mRange:
        print('processing mock: %d' %imock)
        np.random.seed(imock)
        tableMock   =   astTab.Table(data=np.empty((nobj,len(names))),names=names)
        tzmc  =   np.empty(nobj, dtype=float)
        # Monte Carlo z
        r     =   np.random.random(size=nobj)
        for i in range(nobj):
            tzmc[i] =   np.interp(r[i], cdf[i], poz_bins)
        g1n,g2n =   rotCatalog(g1,g2)
        tableMock['raH']    =   ra
        tableMock['decH']   =   dec
        tableMock['raR']    =   raR
        tableMock['decR']   =   decR
        tableMock['g1n']    =   g1n
        tableMock['g2n']    =   g2n
        tableMock['zbest']  =   zBest
        tableMock['ztrue']  =   tzmc
        tableMock.write('HSC-obs/20200328/cats/sim%d.fits' %imock)
        del tableMock
        del g1n,g2n
        del tzmc
        gc.collect()
    return

if __name__ == "__main__":
    main()
