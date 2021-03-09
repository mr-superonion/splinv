# Copyright 20200227 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#

import os
import numpy as np

def mcSample(var,pdf):
    pdf =   pdf.astype(float)
    psp =   pdf.shape
    if len(psp)==1:
        # normalize
        nbin=   len(pdf)
        pdf =   pdf.astype(float)
        pdf /=  np.sum(pdf)

        #cumulative distribution
        cdf =   np.empty(nbin,dtype=float)
        np.cumsum(pdf, out=cdf)

        # Monte Carlo sampling
        r     = np.random.random(size=1)
        s     = np.interp(r, cdf, var)
    elif len(psp)==2:
        # normalize
        nobj, nbin = pdf.shape
        pdf /= np.sum(pdf,axis=1).reshape(nobj, 1)

        #cumulative distribution
        cdf = np.empty(shape=(nobj, nbin), dtype=float)
        np.cumsum(pdf, axis=1, out=cdf)

        # Monte Carlo sampling
        r   =   np.random.random(size=nobj)
        s   =   np.empty(nobj, dtype=float)
        for i in range(nobj):
            s[i] = np.interp(r[i],cdf[i],var)
    return s
