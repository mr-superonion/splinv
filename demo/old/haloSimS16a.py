#!/usr/bin/env python
import os
import hmf
import galsim
import numpy as np
from configparser import ConfigParser
import astropy.table as astTab

def makehaloCat(fdName,parser):
    #columns for haloCat
    #'z_cl','M_200','conc','ra','dec'
    outFname=   './s16aSim/haloCat_%s.csv' %fdName
    if os.path.exists(outFname):
        print('already have haloCat for %s' %fdName)
        return astTab.Table.read(outFname)
    mMin    =   14.5
    mMax    =   16.5
    scale  =   parser.getfloat('transPlane','scale')
    ny     =   parser.getint('transPlane'  ,'ny')
    nx     =   parser.getint('transPlane'  ,'nx')
    xMin   =   parser.getfloat('transPlane','xMin')
    yMin   =   parser.getfloat('transPlane','yMin')
    xMax   =   xMin+nx*scale
    yMax   =   yMin+ny*scale
    yMax2  =   np.pi/2.-yMin*np.pi/180.
    yMin2  =   np.pi/2.-yMax*np.pi/180.
    area   =   nx*np.pi/180.*scale*(np.cos(yMin2)-np.cos(yMax2))
    nlp    =   parser.getint('lensZ','nlp')

    #We need to know halo density mass function in each lens bin
    #lens red shift bin
    zlMin  =   parser.getfloat('lensZ','zlMin')
    zlscale=   parser.getfloat('lensZ','zlscale')
    zlBin  =   np.arange(zlMin,zlMin+zlscale*nlp,zlscale)+zlscale/2.
    hmfun  =   hmf.MassFunction(Mmin=mMin,Mmax=mMax)
    dcmV   =   hmfun.parameter_values['cosmo_model'].differential_comoving_volume(zlBin).value

    mAll   =   []
    zAll   =   []
    conc   =   []
    for i,iz in enumerate(zlBin):
        hmfun.update(z=iz)
        volumn=dcmV[i]*zlscale*area
        nhalo=int(hmfun.ngtm[0]*volumn+0.5)
        zmin=iz-zlscale/2.
        zmax=iz+zlscale/2.
        zAll.extend(np.random.uniform(low=zmin,high=zmax,size=nhalo))
        mAll.extend(hmf.sample_mf(N=nhalo,log_mmin=mMin,z=iz,Mmax=mMax)[0])

    zAll=np.array(zAll)
    mAll=np.array(mAll)
    concAll=6.02*(mAll/1.E13)**(-0.12)*(1.47/(1.+zAll[i]))**(0.16)
    raAll=np.random.uniform(low=xMin,high=xMax,size=len(zAll))
    decAll=np.random.uniform(low=yMin,high=yMax,size=len(zAll))
    cols=(zAll,mAll,concAll,raAll,decAll)
    names=('z_cl','M_200','conc','ra','dec')
    haloTab=astTab.Table(cols,names=names)
    haloTab.write(outFname)
    return haloTab


def makeShapeCat(fdName,haloCat,parser):
    simSrcName  =   './s16aPre/%s_RG_mock.fits' %(fdName)
    if not os.path.exists(simSrcName):
        print('do not have mock catalog for %s' %fdName)
        return
    if parser.has_option('sourceZ','zname'):
        zname   =   parser.get('sourceZ','zname')
    else:
        zname   =   'z'
    if parser.has_option('transPlane','raname'):
        raname  =   parser.get('transPlane','raname')
    else:
        raname  =   'ra'
    if parser.has_option('transPlane','decname'):
        decname =   parser.get('transPlane','decname')
    else:
        decname=   'dec'
    print('begin simulate catalog')
    simSrc  =   astTab.Table.read(simSrcName)
    ras     =   simSrc[raname]
    decs    =   simSrc[decname]
    zs      =   simSrc[zname]
    print('Getting shear field for each galaxies')
    for hh in haloCat:
        #positionD should be in arcsec
        pos_cl  =   galsim.PositionD(hh['ra']*3600,hh['dec']*3600.)
        halo    =   galsim.nfw_halo.NFWHalo(mass= hh['M_200'],
                conc=hh['conc'], redshift= hh['z_cl'],
                halo_pos=pos_cl ,omega_m= 0.3,
                omega_lam= 0.7)
        g1s,g2s=    halo.getShear(pos=(ras,decs),z_s=zs,units = "degree",reduced=False)
    outSrc          =   astTab.Table()
    outSrc['ra']    =   ras 
    outSrc['dec']   =   decs
    outSrc['gamma1']=   g1s
    outSrc['gamma2']=   g2s

    print('Adding shape noise')
    for isim in range(100):
        znameU      =   zname+'_%d' %isim
        outSrc['g1_%d' %isim]=  outSrc['gamma1']+simSrc['g1_%d' %isim]
        outSrc['g2_%d' %isim]=  outSrc['gamma2']+simSrc['g2_%d' %isim]
        outSrc['z_%d'  %isim]=  simSrc[znameU]
    outSrc.write('./s16aSim/mockSrcHalo_%s.fits' %fdName)
    return
    

if __name__=='__main__':
    fieldList   =  np.load('./fieldInfo.npy',allow_pickle=True).item().keys() 
    for fdName in fieldList:
        parser =   ConfigParser()
        configDir   =   's16a3D/pix-0.05/nframe-3/lambda-3.5/' 
        configName  =   os.path.join(configDir,'config_lbd3.5_%s.ini' %fdName)
        parser.read(configName)
        haloCat =   makehaloCat(fdName,parser)
        makeShapeCat(fdName,haloCat,parser)
