#!/usr/bin/env python
import os
import numpy as np
from configparser import ConfigParser


def addInfoSparse(parser,lbd):
    #sparse
    doDebug=    'no'
    nframe =    4
    nMax   =    1
    maxR   =    5

    parser['sparse']={  'doDebug':'%s'%doDebug,
                        'lbd'   :'%s' %lbd,
                        'nframe':'%s' %nframe,
                        'nMax'  :'%s' %nMax,
                        'maxR'  :'%s' %maxR}
    #transverse plane
    unit        =   'arcmin'
    xMin        =   -24.    #(arcmin)
    yMin        =   -24.    #(arcmin)
    scale       =   0.25    #(arcmin/pix)
    ny          =   192     #pixels
    nx          =   192     #pixels

    parser['transPlane']={'unit':'%s'  %unit,
                         'xMin':'%s'  %xMin,
                         'yMin' :'%s'  %yMin,
                         'scale':'%s'  %scale,
                         'ny'   :'%s'  %ny,
                         'nx'   :'%s'  %nx}

    #lens z axis
    zlMin       =   0.01
    zlscale     =   0.05
    nlp         =   20
    parser['lensZ']={'zlMin'    :   '%s'  %zlMin,
                     'zlscale'  :   '%s'  %zlscale,
                     'nlp'      :   '%s'  %nlp}

    #source z axis
    nz          =   8
    if nz!=1:
        zMin        =   0.05
        zscale      =   0.25
    else:
        zMin    =   0.
        zscale  =   4.
    parser['sourceZ']={ 'zMin'  :   '%s'  %zMin,
                        'zscale':   '%s'  %zscale,
                        'nz'    :   '%s'  %nz}
    return parser


if __name__=='__main__':
    simDir  =   'simulation9'
    if not os.path.exists(simDir):
        os.mkdir(simDir)
    lbd     =   8
    for im,logM in enumerate(np.arange(13,16,0.2)):
        for iz,z_cl in enumerate(np.arange(0.02,1.04,0.04)):
            parser      =   ConfigParser()
            # cosmology
            omega_m     =   0.3
            omega_l     =   0.7
            h_cos       =   0.7
            parser['cosmology']={'omega_m':'%s'  %omega_m,
                                'omega_l' :'%s'  %omega_l, 
                                'h_cos'   :'%s'  %h_cos}
            # sources 
            size        =   32              #(arcmin)
            ns_per_arcmin=  40              
            var_gErr    =   0.25
            parser['sources']={'size'          :'%s'  %size,
                               'ns_per_arcmin' :'%s'  %ns_per_arcmin, 
                               'var_gErr'      :'%s'  %var_gErr}
            # lens 
            x_cl        =   0.*60.          #arcsec
            y_cl        =   0.*60.          #arcsec
            M_200       =   10**logM*h_cos  #(M_sun/h)

            parser['lens']={
                            'm_id'      :'%s'  %im,
                            'z_id'      :'%s'  %iz,
                            'z_cl'      :'%s'  %z_cl,
                            'x_cl'      :'%s'  %x_cl, 
                            'y_cl'      :'%s'  %y_cl, 
                            'M_200'     :'%s'  %M_200}
            parser  =   addInfoSparse(parser,lbd)
            configName  =   os.path.join(simDir,'config_lbd%d_m%d_z%d.ini' %(lbd,im,iz))
            with open(configName, 'w') as configfile:
                parser.write(configfile)
