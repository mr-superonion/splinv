#!/usr/bin/env python
import os
import itertools
import astropy.table as astTab
import numpy as np
from multiprocessing import Pool



def prepareHSCField16(fieldname):
    if fieldname!='VVDS':
        return
    onameRG =   './s16aPre/%s_RG.fits' %(fieldname)
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
    return


def main():
    pool    =   Pool(1)
    fields  =   np.load('./dr1FieldTract.npy').item().keys() 
    pool.map(prepareHSCField16,fields)
    return

if __name__ == '__main__':
    main()
