import numpy as np
from scipy import ndimage as ndi

def local_maxima_3D(data,npixt=2,npixp=1,threshold=1.):
    """Detects local maxima in a 3D array

    Parameters
    ---------
    data : 3d ndarray
    npixt : int
        How many points on each dimension of transverse plane to use for the
        comparison

    npixp : int
        How many points on the line-of-sight dimension to use for the
        comparison

    Returns
    -------
    coords : ndarray
        coordinates of the local maxima
    values : ndarray
        values of the local maxima
    """
    sizet = 1 + 2 * npixt
    sizep = 1 + 2 * npixp
    footprint = np.ones((sizep, sizet, sizet))
    footprint[npixp, npixt, npixt] = 0
    filtered    =   ndi.maximum_filter(data,footprint=footprint,mode='constant')
    mask_local_maxima = (data > filtered)& (data>threshold)
    coords = np.asarray(np.where(mask_local_maxima)).T

    footprint[npixp, npixt, npixt] = 1
    smoothed    =   ndi.convolve(data,weights=footprint,mode='constant')
    values = smoothed[mask_local_maxima]
    return coords, values

def local_minima_3D(data,npixt=2,npixp=2,threshold=1.):
    """Detects local minima in a 3D array

    Parameters
    ---------
    data : 3d ndarray
    npixt : int
        How many points on each dimension of transverse plane to use for the
        comparison

    npixp : int
        How many points on the line-of-sight dimension to use for the
        comparison

    Returns
    -------
    coords : ndarray
        coordinates of the local minima
    values : ndarray
        values of the local minima
    """
    sizet = 1 + 2 * npixt
    sizep = 1 + 2 * npixp
    footprint = np.ones((sizep, sizet, sizet))
    footprint[npixp, npixt, npixt] = 0

    filtered = ndi.minimum_filter(data, footprint=footprint,mode='constant')
    mask_local_minima = (data < filtered)& (data<-threshold)
    coords = np.asarray(np.where(mask_local_minima)).T
    values = data[mask_local_minima]
    return coords, values
