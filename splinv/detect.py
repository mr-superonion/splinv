# Copyright 20211226 Xiangchong Li.
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
import numpy as np
from scipy import ndimage as ndi

def local_maxima_3D(data,npixt=2,npixp=1,threshold=1.):
    """Detects local maxima in a 3D array

    Parameters:
    data :  [3D ndarray]
    npixt:  How many points on each dimension of transverse plane to use for the
            comparison [int]

    npixp : How many points on the line-of-sight dimension to use for the
            comparison [int]

    Returns:
        coords: coordinates of the local maxima [ndarray]
        values: values of the local maxima [ndarray]
    """
    sizet = 1 + 2 * npixt
    sizep = 1 + 2 * npixp
    footprint = np.ones((sizep, sizet, sizet))
    footprint[npixp, npixt, npixt] = 0
    filtered    =   ndi.maximum_filter(data,footprint=footprint,mode='constant')
    mask_local_maxima = (data > filtered)& (data>threshold)
    coords = np.int_(np.asarray(np.where(mask_local_maxima)).T)

    footprint[npixp, npixt, npixt] = 1
    smoothed    =   ndi.convolve(data,weights=footprint,mode='constant')
    values = smoothed[mask_local_maxima]
    return coords, values

def local_minima_3D(data,npixt=2,npixp=2,threshold=1.):
    """Detects local minima in a 3D array

    Parameters:
    data :  [3D ndarray]
    npixt:  How many points on each dimension of transverse plane to use for
            the comparison [int]
    npixp:  How many points on the line-of-sight dimension to use for the
            comparison [int]

    Returns:
        coords : coordinates of the local minima [ndarray]
        values : values of the local minima [ndarray]
    """
    sizet = 1 + 2 * npixt
    sizep = 1 + 2 * npixp
    footprint = np.ones((sizep, sizet, sizet))
    footprint[npixp, npixt, npixt] = 0

    filtered = ndi.minimum_filter(data, footprint=footprint,mode='constant')
    mask_local_minima = (data < filtered)& (data<-threshold)
    coords = np.int_(np.asarray(np.where(mask_local_minima)).T)
    values = data[mask_local_minima]
    return coords, values
