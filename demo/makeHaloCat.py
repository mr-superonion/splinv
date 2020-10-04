#!/usr/bin/env python

import haloSim
import numpy as np
import astropy.table as astTab

omega_m=0.315   # Planck 2018
#omega_m=0.3    # Toy model

def main():
    log_m_array=np.linspace(13.3,15.0,8) # [M_solar/h]
    z_array=np.linspace(0.05,0.8,8)
    names=('iz','im','zh','log10_M200','conc','rs_arcmin')
    data=[]
    for iz,zh in enumerate(z_array):
        for im,logm in enumerate(log_m_array):
            z_h =   zh+np.random.uniform(-0.02,0.02)
            log_m=  logm+np.random.uniform(-0.05,0.05)

            M_200=  10.**(log_m) # in unit of M_solar/h
            conc =  6.02*(M_200/1.E13)**(-0.12)*(1.47/(1.+z_h))**(0.16)
            halo =  haloSim.nfw_lensTJ03(mass=M_200,conc=conc,redshift=z_h,\
                    ra=0.,dec=0.,omega_m=omega_m)
            rs_amin=halo.rs_arcsec/60.
            data.append((iz,im,z_h,log_m,conc,rs_amin))
    tabOut=astTab.Table(rows=data,names=names)
    tabOut.write('haloCat-202010032144.csv')
    return

if __name__ == "__main__":
    main()
