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
import json
import cosmology
import numpy as np

import astropy.io.fits as pyfits
from configparser import ConfigParser

def zMeanBin(zMin,dz,nz):
    return np.arange(zMin,zMin+dz*nz,dz)+dz/2.

def soft_thresholding(dum,thresholds):
    """
    Soft-Threshold Function

    Parameters:
    ----------
    dum:    array to panelize
    thresholds: panelization threshold
    """
    return np.sign(dum)*np.maximum(np.abs(dum)-thresholds,0.)

def soft_thresholding_nn(dum,thresholds):
    """
    Non-negative Soft-Threshold Function

    Parameters:
    ----------
    dum:    array to panelize
    thresholds: panelization threshold

    """
    return np.maximum(dum-thresholds,0.)

def firm_thresholding(dum,thresholds):
    """
    Firm thresholding used by Glimpse3D-2013

    Parameters:
    ----------
    dum:    array to panelize
    thresholds: panelization threshold

    """
    mask    =   (np.abs(dum)<= thresholds)
    dum[mask]=  0.
    mask    =   (np.abs(dum)>thresholds)
    mask    =   mask&(np.abs(dum)<= 2*thresholds)
    dum[mask]=  np.sign(dum[mask])*(2*np.abs(dum[mask])-thresholds[mask])
    return dum

class massmapSparsityTaskNew():
    def __init__(self,parser):
        # Regularization
        self.lbd        =   parser.getfloat('sparse','lbd') #   For LASSO
        if parser.has_option('sparse','tau'):               #   For Total Square Variance
            self.tau    =   parser.getfloat('sparse','tau')
        else:
            self.tau    =   0.
        # Dictionary
        self.nframe     =   parser.getint('sparse','nframe')

        # Transverse plane
        self.nx         =   parser.getint('transPlane','nx')
        self.ny         =   parser.getint('transPlane','ny')
        # Lens redshift axis
        self.nlp        =   parser.getint('lensZ','nlp')
        if self.nlp<=1:
            # 2D case
            self.zlMin  =   0.
            self.zlscale=   1.
        else:
            self.zlMin  =   parser.getfloat('lensZ','zlMin')
            self.zlscale=   parser.getfloat('lensZ','zlscale')
        self.zlBin      =   zMeanBin(self.zlMin,self.zlscale,self.nlp)

        # Source z axis
        self.nz     =   parser.getint('sourceZ','nz')
        if self.nz<=1:
            # 2D case
            assert self.nlp<=1
            self.zMin   =   0.01
            self.zscale =   2.5
            self.zsBin  =   zMeanBin(self.zMin,self.zscale,self.nz)
        else:
            zbound      =   np.array(json.loads(parser.get('sourceZ','zbound')))
            self.zsBin  =   (zbound[:-1]+zbound[1:])/2.

        self.shapeS     =   (self.nz,self.ny,self.nx)
        self.shapeL     =   (self.nlp,self.ny,self.nx)
        self.shapeA     =   (self.nlp,self.nframe,self.ny,self.nx)

        dicname     =   parser.get('sparse','dicname')
        from halolet import nfwShearlet2D
        self.dict2D =   nfwShearlet2D(parser)

        # Read pixelized sigma map for shear
        sigfname    =   parser.get('prepare','sigmafname')
        self.sigmaS =   pyfits.getdata(sigfname)
        assert self.sigmaS.shape  ==   self.shapeS, \
            'load wrong pixelized std, shape map shape: (%d,%d,%d)' %self.sigmaS.shape
        # Mask in the shear observation space
        self.maskS   =   (self.sigmaS>=1.e-4)
        self.sigmaSInv=  np.zeros(self.shapeS)
        self.sigmaSInv[self.maskS]=  1./self.sigmaS[self.maskS]
        self.read_lens_kernel(parser)

        # Estimate diagonal elements of the chi2 operator
        self.fast_chi2diagonal_est()
        # weight for normalization of effective column vectors
        self._w     =   1./np.sqrt(self.diagonal+4.*self.tau+1.e-15)

        # Estimate sigma map for alpha
        self.prox_sigmaA()

        self.clear_all()
        # Determine Step Size: mu
        if parser.has_option('sparse','mu'):
            self.mu =   parser.getfloat('sparse','mu')
            if self.mu <0:
                self.determine_step_size()
        else:
            self.determine_step_size()

        # read the pixelized shear,mask Note: this should be done after
        # determine step size, since determine_step_size requires self.shearR
        # to be zero
        self.read_pixel_result(parser)

        # Do debug?
        # if parser.has_option('sparse','debugList'):
        #     self.debugList  =   np.array(json.loads(parser.get('sparse','debugList')))
        #     self.debugRatios=[]
        #     self.debugDeltas=[]
        #     self.debugAlphas=[]
        # else:
        #     debugList   =   []
        return

    def clear_all(self):
        # Clear results
        self.alphaR =   np.zeros(self.shapeA)   # alpha
        self.deltaR =   np.zeros(self.shapeL)   # delta
        self.shearRRes   = np.zeros(self.shapeS)# shear residuals
        self.shearR =   np.zeros(self.shapeS)   # shear
        return

    def read_pixel_result(self,parser):
        """
        Read the pixelized g1,g2

        Parameters:
        ----------
        parser: config parser

        """
        g1fname     =   parser.get('prepare','g1fname')
        g2fname     =   parser.get('prepare','g2fname')
        g1Map       =   pyfits.getdata(g1fname)
        g2Map       =   pyfits.getdata(g2fname)
        assert g1Map.shape  ==   self.shapeS, \
            'load wrong pixelized shear 1, shape should be: (%d,%d,%d)' %self.shapeS
        assert g2Map.shape  ==   self.shapeS, \
            'load wrong pixelized shear 2, shape should be: (%d,%d,%d)' %self.shapeS
        self.shearR =   g1Map+np.complex64(1j)*g2Map # shear
        return

    def read_lens_kernel(self,parser):
        """
        Read the pixelized lensing kernel (normalization not required)

        Parameters:
        ----------
        parser: config parser

        """
        lkfname     =   parser.get('prepare','lkfname')
        self.lensKernel=pyfits.getdata(lkfname)
        assert self.lensKernel.shape  ==   (self.nz,self.nlp), \
            'load wrong lensing kernel, shape should be: (%d,%d)' \
                %(self.nz,self.nlp)
        return

    def main_forward(self,alphaRIn):
        """
        Transform from dictionary space to observational space

        Parameters:
        ----------
        alphaRIn: modes in dictionary space.

        """
        # self._w normalizes the forward operator: A
        alphaRIn    =   alphaRIn*self._w
        shearOut    =   self.dict2D.itransform(alphaRIn)
        shearOut    =   shearOut*(self.maskS.astype(np.int))
        return shearOut

    def chi2_transpose(self,shearRIn):
        """
        Traspose operation on observed map

        Parameters:
        ----------
        shearRIn: input observed map (e.g. opbserved shear map)

        """
        # initializate an empty delta map
        # only keep the E-mode
        alphaRO     =   self.dict2D.itranspose(shearRIn).real
        return alphaRO*self._w

    def gradient_chi2(self,alphaR):
        """
        Gradient operation of Chi2 act on dictionary space

        Parameters:
        ----------
        alphaR: [array]
                the point at which the gradient is calulated

        """
        shearRTmp       =   self.main_forward(alphaR)               #A_{ij} x_j
        self.shearRRes  =   (self.shearR-shearRTmp)*self.sigmaSInv  #y_i-A_{ij} x_j
        return -self.chi2_transpose(self.shearRRes*self.sigmaSInv)  #-A_{i\alpha}(y_i-A_{ij}x_j)

    def gradient_TSV(self,alphaR):
        """
        Calculate the gradient of the Total Square Variance(TSV) component
        finite difference operator

        Parameters:
        ----------
        alphaR: [array]
                the point at which the gradient is calulated

        """
        alphaR  =   alphaR*self._w
        difx    =   np.roll(alphaR,1,axis=-1)
        difx    =   difx-alphaR #D1
        # Transpose of the finite difference operator
        gradx   =   np.roll(difx,-1,axis=-1)
        gradx   =   gradx-difx # (S_1)_{i\alpha} (S1)_{i\alpha} x_\alpha

        # The same for the theta2 direction
        dify    =   np.roll(alphaR,1,axis=-2)
        dify    =   dify-alphaR #D2
        grady   =   np.roll(dify,-1,axis=-2)
        grady   =   grady-dify # (S)_{ij} (S_2)_{i\alpha} x_\alpha

        """
        # for the z direction
        difz    =   np.roll(alphaR,1,axis=-2)
        difz    =   difz+alphaR #D2
        gradz   =   np.roll(difz,-1,axis=-2)
        gradz   =   gradz+difz  #(S)_{ij} (S_2)_{i\alpha} x_\alpha
        """
        gradz   =   0.
        return (gradx+grady+gradz)*self.tau*self._w

    def gradient_Quad(self,alphaR):
        """
        calculate the gradient of the second order component in the loss
        function wihch includes chi2 components and other second order
        regularization terms (e.g. TSV, Ridge).

        Parameters:
        ---------
        alphaR: [array]
                the point at which the gradient is calulated

        """
        gCh2    =   self.gradient_chi2(alphaR)
        if np.max(self.tau)>0.:
            gTSV    =   self.gradient_TSV(alphaR)
        else:
            gTSV    =   0.
        return gCh2+gTSV

    def fast_chi2diagonal_est(self):
        """
        Estimate the diagonal elements of the Chi2 transform matrix
        """
        # The space is weighted by var^{-2}
        spaceF  =   np.fft.fft2((self.sigmaSInv**2.)) #[nz,ny,nx]
        # l2-norm in weighted space
        fun     =   np.conj(self.dict2D.aframes)*self.dict2D.aframes
        fun     =   np.fft.fft2(fun) #[nlp,nframe,ny,nx]
        fun     =   fun[None,:,:,:,:]*spaceF[:,None,None,:,:]
        asquareframe=   np.fft.ifft2(fun).real
        self.diagonal=  np.sum(self.lensKernel[:,:,None,None,None]**2.\
                *asquareframe,axis=0)
        return

    # def fast_chi2diagonal_est(self):
    #     """
    #     Estimate the diagonal elements of the Chi2 transform matrix
    #     """
    #     asquareframe=   np.zeros((self.nz,self.nframe,self.ny,self.nx))
    #     for iz in range(self.nz):
    #         spaceF   =   np.fft.fft2((self.sigmaSInv[iz]**2.))
    #         for iframe in range(self.nframe):
    #             fun=np.conj(self.dict2D.aframes[iz,iframe])*self.dict2D.aframes[iz,iframe]
    #             asquareframe[iz,iframe,:,:]=np.fft.ifft2(np.fft.fft2(fun)*spaceF).real

    #     self.diagonal=  np.sum(self.lensKernel[:,:,None,None,None]**2.\
    #             *asquareframe[:,None,:,:,:],axis=0)
    #     return

    def determine_step_size(self):
        norm        =   0.
        for irun in range(1000):
            # generate a normalized random vector
            np.random.seed(irun)
            alphaTmp=   np.random.randn(self.nlp,self.nframe,self.ny,self.nx)
            alphaTmp=   alphaTmp+np.random.randn()
            normTmp =   np.sqrt(np.sum(alphaTmp**2.))
            alphaTmp=   alphaTmp/normTmp
            # apply the transform matrix to the vector
            alphaTmp2=  self.gradient_Quad(alphaTmp)
            normTmp2=   np.sqrt(np.sum(alphaTmp2**2.))
            if normTmp2>norm:
                norm=normTmp2
        self.mu =   1./norm/3.
        return

    def prox_sigmaA(self):
        """
        Calculate stds of the paramters Note that the std should be all 1 since
        we normalize the projectors.  However, we set some stds to +infty
        (masked out)
        """
        niter   =   100
        # A_i\alpha n_i
        outData     =   np.zeros(self.shapeA)
        for irun in range(niter):
            np.random.seed(irun)
            # Here the sigmaS is not the per-component sigma.
            # While, we use the std for g1+ig2, which should be sqrt(2) times of
            # the per component std.
            # Since in the transpose operation, we only keep the real part of
            # alpha field, such operation makes the alpha's std to 1/sqrt(2) of
            # the ture one. Therefore, the final estimation is correct.
            g1Sim   =   np.random.randn(self.nz,self.ny,self.nx)*self.sigmaS
            g2Sim   =   np.random.randn(self.nz,self.ny,self.nx)*self.sigmaS
            shearSim=   (g1Sim+np.complex64(1j)*g2Sim)*self.sigmaSInv**2.
            alphaRSim=  -self.chi2_transpose(shearSim) # Real field
            outData +=  alphaRSim**2.

        # masked region is assigned with the maximum of the std in this frame
        # and lens redshift plane
        # TODO: accelerate
        maskL =   np.all(self.maskS,axis=0)
        for izlp in range(self.nlp):
            for iframe in range(self.nframe):
                outData[izlp,iframe][~maskL]=np.max(outData[izlp,iframe])
        # Calculate noise std
        self.sigmaA =   np.sqrt(outData/niter)

        # mask the parameter field close to the boundary of the survey set the
        # stds of these regions to +infty
        for izl in range(self.nlp):
            for iframe in range(self.nframe):
                thres=np.max(self.diagonal[izl,iframe].flatten())/10.
                maskLP= self.diagonal[izl,iframe]>thres
                self.sigmaA[izl,iframe][~maskLP]=1e12
        return

    def reconstruct(self):
        """
        Reconstruct the delta field from alpha'
        """
        # reweight back to True unweighted alpha
        alphaRT     =   self.alphaR.copy()*self._w
        # transform from dictionary field to delta field
        self.deltaR =   self.dict2D.itransformInter(alphaRT).real
        return

    def adaptive_lasso_weight(self,gamma=1,sm_scale=0.25):
        """Calculate adaptive weight

        Parameters:
        -----------
        gamma:     power of the root-n consistent (preliminary)
                    estimation
        sm_scale:  top-hat smoothing scale for the root-n
                    consistent estimation [Mpc/h]
        """
        # if self.nframe==1 and sm_scale>1e-4:
        #     # Smoothing scale in arcmin
        #     rsmth0=np.zeros(self.nlp,dtype=int)
        #     for iz,zh in enumerate(self.zlBin):
        #         rsmth0[iz]=(np.round(sm_scale/self.cosmo.Dc(0.,zh)*60*180./np.pi))

        #     p   =   np.zeros(self.shapeA)
        #     for izl in range(self.nlp):
        #         rsmth   =   rsmth0[izl]
        #         for jsh in range(-rsmth,rsmth+1):
        #             # only smooth the point mass frame
        #             dif    =   np.roll(self.alphaR[izl,0],jsh,axis=-2)
        #             for ish in range(-rsmth,rsmth+1):
        #                 dif2=  np.roll(dif,ish,axis=-1)
        #                 p[izl,0] += dif2/(2.*rsmth+1.)#**2.
        #     p   =   np.abs(p)
        # else:
        p   =   np.abs(self.alphaR)/self.lbd

        # threshold(for value close to zero)
        thres_adp=  1./1e12
        mask=   (p**gamma>thres_adp)

        # weight estimation
        w       =   np.zeros(self.shapeA)
        w[mask] =   1./(p[mask])**(gamma)
        w[~mask]=   1./thres_adp
        return w

    def fista_gradient_descent(self,niter,w=1.):
        # The thresholds
        thresholds =   self.lbd*self.sigmaA*self.mu*w
        # FISTA algorithms
        tn      =   1.
        Xp0     =   self.alphaR
        for irun in range(niter):
            # (.real means no B-mode)
            dalphaR =   -self.mu*self.gradient_Quad(self.alphaR).real
            Xp1 =   self.alphaR+dalphaR
            Xp1 =   soft_thresholding_nn(Xp1,thresholds)
            #Xp1 =   soft_thresholding(Xp1,thresholds)
            tnTmp= (1.+np.sqrt(1.+4*tn**2.))/2.
            ratio= (tn-1.)/tnTmp
            self.alphaR=Xp1+(ratio*(Xp1-Xp0))
            tn  =   tnTmp
            Xp0 =   Xp1
        return

    def process(self,niter=1000):
        self.fista_gradient_descent(niter)
        self.reconstruct()
        return
