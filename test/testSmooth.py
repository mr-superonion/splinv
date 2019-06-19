import kmapUtilities as utilities
import numpy as np
import pyfits

ny  =   48
nx  =   48
a   =   np.zeros((ny,nx))
a[24,24]=   1. 
sigma   =   5.
bFou    =   np.zeros((ny,nx))
bReal   =   np.zeros((ny,nx))

for j in range(ny):
    jy  =   (j+ny//2)%ny-ny//2
    jy  =   2.*np.pi*jy/ny
    for i in range(nx):
        ix  =   (i+nx//2)%nx-nx//2
        ix  =   2.*np.pi*ix/nx
        bFou[j,i] =   np.exp(-(ix**2.+jy**2.)*sigma**2./2.)

for j in range(ny):
    jy  =   j-ny//2
    for i in range(nx):
        ix  =   i-nx//2
        bReal[j,i] =   1./sigma**2./2./np.pi*np.exp(-(ix**2.+jy**2.)/sigma**2./2.)

realGaus2   =   utilities.smoothGaus(a,sigma)

pyfits.writeto('fourGaus.fits',bFou)
pyfits.writeto('realGaus.fits',np.fft.fftshift(np.fft.ifft2(bFou).real))
pyfits.writeto('realGaus0.fits',bReal)
pyfits.writeto('realGaus2.fits',realGaus2)
