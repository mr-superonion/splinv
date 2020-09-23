import numpy as np
from scipy import ndimage as ndi

def local_maxima_3D(data, ordert=2, orderp=1, threshold=1.):
    """Detects local maxima in a 3D array

    Parameters
    ---------
    data : 3d ndarray
    ordert : int
        How many points on each dimension of transverse plane to use for the comparison

    orderp : int
        How many points on the line-of-sight dimension to use for the comparison

    Returns
    -------
    coords : ndarray
        coordinates of the local maxima
    values : ndarray
        values of the local maxima
    """
    sizet = 1 + 2 * ordert
    sizep = 1 + 2 * orderp
    footprint = np.ones((sizep, sizet, sizet))
    footprint[orderp, ordert, ordert] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint,mode='mirror')
    mask_local_maxima = (data > filtered)& (data>threshold)
    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]
    return coords, values

def local_minima_3D(data, ordert=2, orderp=1, threshold=0.5):
    """Detects local minima in a 3D array

    Parameters
    ---------
    data : 3d ndarray
    ordert : int
        How many points on each dimension of transverse plane to use for the comparison

    orderp : int
        How many points on the line-of-sight dimension to use for the comparison

    Returns
    -------
    coords : ndarray
        coordinates of the local maxima
    values : ndarray
        values of the local maxima
    """
    sizet = 1 + 2 * ordert
    sizep = 1 + 2 * orderp
    footprint = np.ones((sizep, sizet, sizet))
    footprint[orderp, ordert, ordert] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint,mode='mirror')
    mask_local_maxima = (data > filtered)& (data<-threshold)
    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]
    return coords, values
