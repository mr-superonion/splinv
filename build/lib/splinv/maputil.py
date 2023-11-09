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

def GausAtom(sigma,ny,nx=None,fou=True,lnorm=2.):
    """
    Normalized Gaussian in a postage stamp
    Parameters:
        sigma:  std of Gaussian function; [float]
        ny:     number of pixel y directions; [int]
        nx:     number of pixel for x; [int, default=None]
        fou:    in Fourier/configuration space [bool, ture/false]
        lnorm:  normalized by lp norm [float]
    """
    if nx is None:
        nx  =   ny
    if sigma>0.01:
        x,y =   np.meshgrid(np.fft.fftfreq(nx),np.fft.fftfreq(ny))
        norm=   1. # an initialization
        if fou:
            # Fourier space
            x  *=   (2*np.pi);y*=(2*np.pi)
            rT  =   np.sqrt(x**2+y**2)
            fun =   np.exp(-(rT*sigma)**2./2.)
            if lnorm>0.:
                norm=   (np.sum(fun**lnorm)/(nx*ny))**(1./lnorm)
        else:
            # Configuration space
            x  *=   (nx);y*=(ny)
            rT  =   np.sqrt(x**2+y**2)
            fun =   1./np.sqrt(2.*np.pi)/sigma*np.exp(-(rT/sigma)**2./2.)
            if lnorm>0.:
                norm=   (np.sum(fun**lnorm))**(1./lnorm)
        return  fun/norm
    else:
        if fou:
            return np.ones((ny,nx))
        else:
            out =   np.zeros((ny,nx))
            out[0,0]=1
            return out

def TophatAtom(width,ny,nx=None,fou=True,lnorm=-1):
    """
    Normalized top-hat atom in a postage stamp
    Parameters:
        width:  width of top-hat function(in unit of pixel)
        ny:     number of pixel y directions; [int]
        nx:     number of pixel for x; [int, default=None]
        fou:    in Fourier/configuration space [bool, ture/false]
        lnorm:  normalized by lp norm [float]
    """
    if nx is None:
        nx=ny
    assert width>0.9 and width<nx-4 and width<ny-4
    width   =   int(width+0.5)
    norm    =   1.
    if fou:
        x,y =   np.meshgrid(np.fft.fftfreq(nx),np.fft.fftfreq(ny))
        x  *=   (2*np.pi);y*=(2*np.pi)
        funx=   np.divide(np.sin(width*x/2),(width*x/2),out=np.ones_like(x), where=x!=0)
        funy=   np.divide(np.sin(width*y/2),(width*y/2),out=np.ones_like(y), where=y!=0)
        fun =   funx*funy
        if lnorm>0.:
            norm=   (np.sum(fun**lnorm)/(nx*ny))**(1./lnorm)
    else:
        sx=(nx-width)//2;sy=(ny-width)//2
        fun=np.zeros((ny,nx))
        fun[sy+1:sy+width,sx+1:sx+width]=1./width**2.
    return  fun/norm
