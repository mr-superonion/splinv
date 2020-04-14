#!/usr/bin/env python
import os
import fitsio
import itertools
import astropy.table as astTab
import numpy as np
from multiprocessing import Pool

def rotCatalog(e1, e2, phi=None):
    if phi  ==  None:
        phi = 2.0 * np.pi * np.random.rand(len(e1))
    cs = np.cos(phi)
    ss = np.sin(phi)
    e1_rot  = e1 * cs + e2 * ss
    e2_rot  = (-1.0) * e1 * ss + e2 * cs
    return e1_rot, e2_rot

def prepareHSCField16(fTone):
    field,tract =   fTone
    tract   =   tract.split('.')[0]
    print('processing field: %s, tract: %s' %(field,tract))
    if field    ==  'AEGIS':
        return
    onameRG     =   './s16aPreZPDF/%s_%s_RG.fits' %(field,tract)
    if not os.path.exists(onameRG):
        colnames=   ['object_id','g1','g2','ra','dec','best_z','best_conf','best_std']
        fnameRG =   '/work/xiangchong.li/work/S16AStandard/S16AStandardCalibrated/tract/%s.fits' %(tract)
        if not os.path.exists(fnameRG):
            return
        dataRG  =   astTab.Table.read(fnameRG)

        obj_id  =   dataRG['object_id']
        ra      =   dataRG['coord_ra']*180./np.pi
        dec     =   dataRG['coord_dec']*180./np.pi
        e1      =   dataRG['ishape_hsm_regauss_e1']
        e2      =   dataRG['ishape_hsm_regauss_e2']
        m_b     =   dataRG['ishape_hsm_regauss_derived_shear_bias_m']
        c1_b    =   dataRG['ishape_hsm_regauss_derived_shear_bias_c1']
        c2_b    =   dataRG['ishape_hsm_regauss_derived_shear_bias_c2']
        weight  =   dataRG['ishape_hsm_regauss_derived_shape_weight']
        w_A     =   np.sum(weight)
        m_bA    =   np.sum(m_b*weight)/w_A
        Res_A   =   1.-np.sum(dataRG['ishape_hsm_regauss_derived_rms_e']**2.*weight)/w_A
        g1      =   1./(1.+m_bA)*(e1/2./Res_A-c1_b)
        g2      =   1./(1.+m_bA)*(e2/2./Res_A-c2_b)
        best_z  =   dataRG['mlz_photoz_best']
        best_conf=   dataRG['mlz_photoz_conf_best']
        best_std=   dataRG['mlz_photoz_std_best']
        tableNew=   astTab.Table(data=[obj_id,g1,g2,ra,dec,best_z,best_conf,best_std],names=colnames)
        tableNew.write(onameRG)
    else:
        tableNew=   astTab.Table.read(onameRG)
    pzBinFname  =   '/work/xiangchong.li/work/S16AStandard/S16A_pz_pdf/mlz/target_wide_s16a_xmmlss_9493.0.P.fits'
    cdf_z   =   fitsio.read(pzBinFname,ext=2)['BINS']
    fnamePZ =   '/work/xiangchong.li/work/S16AStandard/S16AStandardCalibrated/tract/%s_pofz.fits' %(tract)
    dataPZ  =   astTab.Table.read(fnamePZ)
    mnameRG =   './s16aPreZPDF/%s_%s_RG_mock.fits' %(field,tract)
    tableMock   =   astTab.Table()
    tableMock['ra']  =   tableNew['ra']
    tableMock['dec'] =   tableNew['dec']
    tableMock['best_z']     =   tableNew['best_z']
    tableMock['best_conf']  =   tableNew['best_conf']
    tableMock['best_std']   =   tableNew['best_std']
    pdf         =   dataPZ['PDF']

    nobj, nbin = pdf.shape
    pdf = pdf.astype(float)
    pdf /= np.sum(pdf,axis=1).reshape(nobj, 1)

    cdf = np.empty(shape=(nobj, nbin), dtype=float)
    np.cumsum(pdf, axis=1, out=cdf)
    for imock in range(100):
        tzmc  =   np.empty(nobj, dtype=float)
        np.random.seed(imock)
        # Monte Carlo z
        r     =   np.random.random(size=nobj)
        for i in range(nobj):
            tzmc[i] = np.interp(r[i], cdf[i], cdf_z)
        tableMock['z_%d' %imock]    =   tzmc
        g1Sim,g2Sim =   rotCatalog(tableNew['g1'],tableNew['g2'])
        tableMock['g1_%d' %imock]   =   g1Sim
        tableMock['g2_%d' %imock]   =   g2Sim
    tableMock.write(mnameRG,overwrite=True)
    return

def main():
    fTinfo  =   np.load('/work/xiangchong.li/work/S16AFPFS/dr1FieldTract.npy').item()
    fields  =   fTinfo.keys()
    fTlist  =   []
    for fd in fields:
        for tt in fTinfo[fd]:
            fTlist.append((fd,tt))
    pool    =   Pool(10)
    pool.map(prepareHSCField16,fTlist)
    return

if __name__ == '__main__':
    main()
