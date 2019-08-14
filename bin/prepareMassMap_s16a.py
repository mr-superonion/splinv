#!/usr/bin/env python
import os
import itertools
import astropy.table as astTab
import numpy as np
from multiprocessing import Pool

def rotCatalog(e1, e2, phi=None):
    if phi  ==  None:
        phi = 2.0 * np.pi * np.random.rand(len(e1))
    cs = np.cos(phi)
    ss = np.sin(phi)
    e1_rot = e1 * cs + e2 * ss
    e2_rot = (-1.0) * e1 * ss + e2 * cs
    return e1_rot, e2_rot

def prepareHSCField16(fieldname):
    if fieldname    !=  'WIDE12H':
        return
    onameRG = './s16aPre2D/%s_RG.fits' %(fieldname)
    if not os.path.exists(onameRG):
        colnames=   ['object_id','g1','g2','ra','dec']
        fnameRG =   '/work/xiangchong.li/work/S16AStandard/S16AStandardCalibrated/field/%s_calibrated.fits' %(fieldname)
        fnamePZ =   '/work/xiangchong.li/work/S16AStandard/S16AStandardV2/field/%s_pz.fits' %(fieldname)
        dataRG  =   astTab.Table.read(fnameRG)
        dataPZ  =   astTab.Table.read(fnamePZ)['object_id','mlz_photoz_best','mlz_photoz_std_best','mlz_photoz_conf_best']

        obj_id  =   dataRG['object_id']
        ra      =   dataRG['ira']
        dec     =   dataRG['idec']
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
        tableNew=   astTab.Table(data=[obj_id,g1,g2,ra,dec],names=colnames)
        tableNew=   astTab.join(tableNew,dataPZ,keys='object_id')
        tableNew.write(onameRG)
    else:
        tableNew=   astTab.Table.read(onameRG)
    mnameRG =   './s16aPre2D/%s_RG_mock.fits' %(fieldname)
    tableMock   =   astTab.Table()
    for imock in range(100):
        np.random.seed(imock)
        rnd     =   np.random.randn(len(tableNew))
        zmock   =   tableNew['mlz_photoz_best']+rnd*tableNew['mlz_photoz_std_best'] 
        tableMock['mlz_photoz_best_%d' %imock]  =   zmock
        g1Sim,g2Sim =   rotCatalog(tableNew['g1'],tableNew['g2'])
        tableMock['g1_%d' %imock]  =   g1Sim
        tableMock['g2_%d' %imock]  =   g2Sim
    tableMock.write(mnameRG)
    return


def main():
    pool    =   Pool(1)
    fields  =   np.load('/work/xiangchong.li/work/S16AFPFS/dr1FieldTract.npy').item().keys() 
    pool.map(prepareHSCField16,fields)
    return

if __name__ == '__main__':
    main()
