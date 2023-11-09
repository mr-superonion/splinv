# Copyright 20220820 Xiangchong Li and Shouzhuo Yang.
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
import json
import numpy as np
from .hmod import nfwShearlet2D
import numba
from numba import float64
from numba.experimental import jitclass

def zMeanBin(zMin,dz,nz):
    return np.arange(zMin,zMin+dz*nz,dz)+dz/2.

@numba.njit
def soft_thresholding(dum,thresholds):
    """
    Soft-Threshold Function

    Parameters:
        dum:    array to panelize
        thresholds: panelization threshold
    """
    return np.sign(dum)*np.maximum(np.abs(dum)-thresholds,0.)

@numba.njit
def soft_thresholding_nn(dum,thresholds):
    """
    Non-negative Soft-Threshold Function

    Parameters:
        dum:    array to panelize
        thresholds: panelization threshold
    """
    return np.maximum(dum-thresholds,0.)

def firm_thresholding(dum,thresholds):
    """Firm thresholding used by Glimpse3D-2013
    Parameters:
        dum:    array to panelize
        thresholds: panelization threshold
    """
    mask    =   (np.abs(dum)<= thresholds)
    dum[mask]=  0.
    mask    =   (np.abs(dum)>thresholds)
    mask    =   mask&(np.abs(dum)<= 2*thresholds)
    dum[mask]=  np.sign(dum[mask])*(2*np.abs(dum[mask])-thresholds[mask])
    return dum

class darkmapper():
    def __init__(self,parser,g1Map,g2Map,gErr,lensKernel):

        # Regularization
        self.lbd        =   parser.getfloat('sparse','lbd') #   For LASSO
        self.lcd        =   parser.getfloat('sparse','lcd') #   For Elastic net
        self.tau        =   parser.getfloat('sparse','tau') #   For TSV
        # Dictionary
        self.nframe     =   parser.getint('sparse','nframe')#   number of frames

        # Transverse plane
        self.ny         =   parser.getint('transPlane','ny')
        self.nx         =   parser.getint('transPlane','nx')
        # Lens redshift axis
        self.nlp        =   parser.getint('lens','nlp')
        if self.nlp<=1:
            # 2D
            self.zlMin  =   0.
            self.zlscale=   1.
        else:
            # 3D
            self.zlMin  =   parser.getfloat('lens','zlMin')
            self.zlscale=   parser.getfloat('lens','zlscale')
        self.zlBin      =   zMeanBin(self.zlMin,self.zlscale,self.nlp)

        # Source z axis
        self.nz     =   parser.getint('sources','nz')
        if self.nz<=1:
            # 2D
            assert self.nlp<=1
            self.zMin   =   0.01
            self.zscale =   2.5
            self.zsBin  =   zMeanBin(self.zMin,self.zscale,self.nz)
        else:
            # 3D
            zbound      =   np.array(json.loads(parser.get('sources','zbound')))
            self.zsBin  =   (zbound[:-1]+zbound[1:])/2.

        self.shapeS     =   (self.nz,self.ny,self.nx)
        self.shapeL     =   (self.nlp,self.ny,self.nx)
        self.shapeA     =   (self.nlp,self.nframe,self.ny,self.nx)

        if not g1Map.shape  ==   self.shapeS:
            raise ValueError("shape of gamma1 map should be: (%d,%d,%d)" %self.shapeS)
        if not g2Map.shape  ==   self.shapeS:
            raise ValueError("shape of gamma2 map should be: (%d,%d,%d)" %self.shapeS)
        if not gErr.shape  ==   self.shapeS:
            raise ValueError("shape of error map should be: (%d,%d,%d)" %self.shapeS)
        if not lensKernel.shape  ==   (self.nz,self.nlp):
            raise ValueError("lensing kernel's shape should be: (%d,%d)" %(self.nz,self.nlp))

        self.modelDict =   nfwShearlet2D(parser,lensKernel)

        # Read pixelized noise std-map for shear
        self.sigmaS =   gErr
        # Mask in the shear observation space
        self.maskS  =   (gErr>=1.e-4)
        self.sigmaSInv  =   np.zeros(self.shapeS)
        self.sigmaSInv[self.maskS]=  1./self.sigmaS[self.maskS]
        self.lensKernel =   lensKernel

        # Estimate diagonal elements of the chi2 operator
        self.fast_chi2diagonal_est()
        # weight for normalization of effective column vectors
        self._w     =   1./np.sqrt(self.diagonal+4.*self.tau+1e-12)

        self.clean_outcomes()
        self.shearR     =   np.zeros(self.shapeS)   # shear

        # Determine Step Size: mu
        self.mu =   parser.getfloat('sparse','mu')
        if self.mu <0:
            self.determine_step_size()

        # Read the pixelized shear and mask. Note: this should be done after
        # determine step size, since determine_step_size requires self.shearR
        # to be zero
        self.shearR =   g1Map+np.complex64(1j)*g2Map # shear
        self.nonNeg =   True
        return

    def clean_outcomes(self):
        """
        # Clear results
        """
        self.alphaR     =   self.maskA              # alpha
        self.deltaR     =   np.zeros(self.shapeL)   # delta
        self.shearRRes  =   np.zeros(self.shapeS)   # shear residuals
        #self.shearProj =   None                    # for Multiplicative updates
        self.diff       =   []
        return

    def main_forward(self,alphaRIn):
        """
        Transform from dictionary space to observational space

        Parameters:
            alphaRIn: modes in dictionary space.

        """
        # self._w normalizes the forward operator: A
        __tmp       =   alphaRIn*self._w
        shearOut    =   self.modelDict.itransform(__tmp)
        # Mask
        shearOut    =   shearOut*(self.maskS.astype(int))
        return shearOut

    def chi2_transpose(self,shearRIn):
        """
        Traspose operation on observed map

        Parameters:
            shearRIn: input observed map (e.g. opbserved shear map)

        """
        # only keep the E-mode
        alphaRO     =   self.modelDict.itranspose(shearRIn).real*self._w*self.maskA
        return alphaRO

    def gradient_chi2(self,alphaR):
        """
        Gradient operation of Chi2 act on dictionary space

        Parameters:
            alphaR: the alpha at which the gradient is calulated

        """
        # Eq (14) of Elastic net (Zou 2005)
        pp  =   self.lcd/(1.+self.lcd)
        # A_{ij} x_j *(1-p)        [weighted A_{ij}]
        shearRTmp       =   self.main_forward(alphaR)*(1-pp)
        # (y_i-(1-p)A_{ij} x_j)/sigma_{ii}    [normalized residual]
        self.shearRRes  =   (self.shearR-shearRTmp)*self.sigmaSInv
        # (-A_{i\alpha}y_i+(1-p)A_{i\alpha}A_{ij}x_j)/sigma^2_{ii}
        out =   -self.chi2_transpose(self.shearRRes*self.sigmaSInv)
        out =   out+alphaR*pp
        return out

    def gradient_TSV(self,alphaR):
        """
        Calculate the gradient of the Total Square Variance(TSV) component
        finite difference operator

        Parameters:
            alphaR: the point at which the gradient is calulated

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
        out     =   (gradx+grady+gradz)*self.tau*self._w
        return out

    def gradient_Quad(self,alphaR):
        """
        calculate the gradient of the second order component in the loss
        function wihch includes chi2 components and other second order
        regularization terms (e.g. TSV, Ridge).

        Parameters:
            alphaR: the point at which the gradient is calulated

        """
        gCh2    =   self.gradient_chi2(alphaR)
        if self.tau>0.:
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
        fun     =   np.conj(self.modelDict.aframes)*self.modelDict.aframes
        fun     =   np.fft.fft2(fun) #[nlp,nframe,ny,nx]
        fun     =   fun[None,:,:,:,:]*spaceF[:,None,None,:,:]
        asquareframe=   np.fft.ifft2(fun).real
        self.diagonal=  np.sum(self.lensKernel[:,:,None,None,None]**2.\
                *asquareframe,axis=0)

        # Determine the mask on the parameter vector
        ## prameters mask during the estimation
        self.maskA  =   np.ones(self.shapeA)
        ## parameters finally kept
        self.maskA2 =   np.ones(self.shapeA)
        for izl in range(self.nlp):
            for iframe in range(self.nframe):
                thres=np.max(self.diagonal[izl,iframe].flatten())/15.
                maskLP= (self.diagonal[izl,iframe]>thres)
                self.maskA[izl,iframe][~maskLP]=0.
                thres2=np.max(self.diagonal[izl,iframe].flatten())/2.
                maskLP2= (self.diagonal[izl,iframe]>thres2)
                self.maskA2[izl,iframe][~maskLP2]=0.
        maskLP      =   np.all(self.maskA,axis=0)
        maskLP2     =   np.all(self.maskA2,axis=0)
        for izl in range(self.nlp):
            self.maskA[izl]=maskLP
            self.maskA2[izl]=maskLP2
        return

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

    def reconstruct(self):
        """
        Reconstruct the delta field from alpha'
        """
        # reweight back to the real unweighted alpha
        alphaRT     =   self.alphaR.copy()*self._w*self.maskA2
        # shrink 1./(1+lcd) if lbd<=0 and lcd>0.
        alphaRT     =   alphaRT/(1.+self.lcd*(self.lbd<=0))
        # transform from dictionary field to delta field
        self.deltaR =   self.modelDict.itransformInter(alphaRT).real
        self.diff   =   np.array(self.diff)
        return

    def adaptive_lasso_weight(self,gamma=1.):
        """
        Calculate adaptive weight for adaptive lasso

        Parameters:
            gamma (float):  power of the root-n consistent (preliminary)
                            estimation [default 1.]
        """
        # sm_scale=0.25
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
        p       =   np.abs(self.alphaR)*self.maskA

        # threshold(for value close to zero)
        thres_adp=  1./1e12
        mask    =   (p**gamma>thres_adp)

        # weight estimation
        w       =   np.zeros(self.shapeA)
        w[mask] =   1./(p[mask])**(gamma)
        w[~mask]=   1./thres_adp
        return w

    def fista_gradient_descent(self,niter,w=1.,tn0=1.):
        """
        FISTA gradient descent solver of loss fucntion
        (Beck & Teboulle 2009)
        Parameters:
            niter (int):      number of iteration
            w (float):        adaptive weight [default: 1.]

        """
        tn  =   tn0
        # The thresholds
        thresholds  =   self.lbd*self.mu*w
        # FISTA algorithms
        Xp0         =   self.alphaR
        self.diff   =   []
        for _ in range(niter):
            # (.real means no B-mode)
            dalphaR =   -self.mu*self.gradient_Quad(self.alphaR).real
            Xp1 =   self.alphaR+dalphaR
            if self.nonNeg:
                Xp1 =   soft_thresholding_nn(Xp1,thresholds)
            else:
                Xp1 =   soft_thresholding(Xp1,thresholds)
            tnTmp= (1.+np.sqrt(1.+4*tn**2.))/2.
            ratio= (tn-1.)/tnTmp
            diff=   Xp1-Xp0
            error=  np.sqrt(np.sum(diff**2.)/np.sum(Xp1**2.))
            self.alphaR=Xp1+(ratio*(diff))
            tn  =   tnTmp
            Xp0 =   Xp1
            self.diff.append(error)
        return
    #@numba.njit
    def fista_gradient_descent_fast(self,niter,w=1.00, tn0=1.00):
        """
        FISTA gradient descent solver of loss fucntion
        (Beck & Teboulle 2009)
        Parameters:
            niter (int):      number of iteration
            w (float):        adaptive weight [default: 1.]
        """
        tn  =  tn0
        # The thresholds
        thresholds  =   float64(self.lbd*self.mu*w)
        # FISTA algorithms
        Xp0         =   self.alphaR
        self.diff   =   np.zeros(niter)
        #error = np.array([])
        for i in range(niter):
            # (.real means no B-mode)
            dalphaR =   -self.mu*self.gradient_Quad(self.alphaR).real
            Xp1 =   self.alphaR+dalphaR
            if self.nonNeg:
                Xp1 =   soft_thresholding_nn(Xp1,thresholds)
            else:
                Xp1 =   soft_thresholding(Xp1,thresholds)
            tnTmp= (1.+np.sqrt(1.+4*tn**2.))/2.
            ratio= (tn-1.)/tnTmp
            diff=   Xp1-Xp0
            error=  np.sqrt(np.sum(diff**2.)/np.sum(Xp1**2.))
            self.alphaR=Xp1+(ratio*(diff))
            tn  =   tnTmp
            Xp0 =   Xp1
            self.diff[i] = error
        return

    def optimized_gradient_descent(self,niter,tn0=1.):
        """
        Optimized gradient descent solver of loss fucntion
        (Kim & Fessier 2017)
        Parameters:
            niter:      number of iteration
        """
        tn          =   tn0
        # OGM algorithms
        Xp0         =   self.alphaR
        self.diff   =   []
        for irun in range(niter):
            # (.real means no B-mode)
            dalphaR =   -self.mu*self.gradient_Quad(self.alphaR).real
            Xp1     =   self.alphaR+dalphaR
            tnTmp   =   (1.+np.sqrt(1.+4.*tn**2.))/2.
            ratio1  =   (tn-1.)/tnTmp
            ratio2  =   tn/tnTmp
            diff1   =   Xp1-Xp0
            diff2   =   Xp1-self.alphaR
            error   =   np.sqrt(np.sum(diff1**2.)/np.sum(Xp1**2.))
            self.alphaR=Xp1+ratio1*diff1+ratio2*diff2
            tn      =   tnTmp
            Xp0     =   Xp1
            self.diff.append(error)
            if irun>200 and error<1e-3:
                break
        tnTmp   =   (1.+np.sqrt(1.+8.*tn**2.))/2.
        ratio1  =   (tn-1.)/tnTmp
        ratio2  =   tn/tnTmp
        diff1   =   Xp1-Xp0
        diff2   =   Xp1-self.alphaR
        error   =   np.sqrt(np.sum(diff1**2.)/np.sum(Xp1**2.))
        self.alphaR=Xp1+ratio1*diff1+ratio2*diff2
        tn      =   tnTmp
        Xp0     =   Xp1
        self.diff.append(error)
        return

    # def multiplicative_update(self,niter,w=1.):
    #     Multiplicative update solver of loss fucntion
    #     (Only works for positive models and positive parameters)

    #     Parameters:
    #         niter:      number of iteration
    #         w:          adaptive weight [default: 1.]
    #     # The thresholds
    #     thresholds  =   self.lbd*w
    #     # A_{i\alpha}y_i/sigma^2_{ii}
    #     if self.shearProj is None:
    #         self.shearProj   =   self.chi2_transpose(self.shearR*self.sigmaSInv**2.)
    #     nominator   =   soft_thresholding_nn(self.shearProj,thresholds)
    #     pp          =   self.lcd/(1+self.lcd)
    #     self.diff   =   []
    #     for irun in range(niter):
    #         # A_{ij} x_j *(1-p)        [weighted A_{ij}]
    #         shearRTmp   =   self.main_forward(self.alphaR)
    #         # ((1-p)A_{i\alpha}A_{ij}x_j)/sigma^2_{ii}
    #         denominator =   self.chi2_transpose(shearRTmp*self.sigmaSInv**2.)
    #         # Add pp*I_{\alphaj} x_j (for Ridge regression)
    #         denominator =   denominator*(1-pp)+self.alphaR*pp
    #         rr      =   nominator/(denominator+1e-12)
    #         alphaR  =   self.alphaR*rr
    #         diff    =   alphaR-self.alphaR
    #         error   =   np.sqrt(np.sum(diff**2.)/np.sum(alphaR**2.))
    #         if irun>200 and error<1e-3:
    #             break
    #         self.diff.append(error)
    #         self.alphaR =   alphaR
    #     return

    # def adaptive_lasso_prior_weight(self,prior):
    #     Calculate adaptive weight using a given prior.
    #     Based on Zou & Li, The Annuals of Statisics 2008,
    #     Vol. 36 No. 4, 1509--1533
    #     (unfinished)

    #     Parameters:
    #         prior:      [nlp,nframe]
    #     p   =   np.abs(self.alphaR)/self.lbd

    #     # threshold(for value close to zero)
    #     thres_adp=  1./1e12
    #     mask=   (p**gamma>thres_adp)

    #     # weight estimation
    #     w       =   np.zeros(self.shapeA)
    #     w[mask] =   1./(p[mask])**(gamma)
    #     w[~mask]=   1./thres_adp
    #     return w

    # def prox_sigmaA(self):
    #     """
    #     Calculate stds of the paramters Note that the std should be all 1 since
    #     we normalize the projectors. We set some stds to +infty
    #     """
    #     niter   =   100
    #     # A_i\alpha n_i
    #     outData     =   np.zeros(self.shapeA)
    #     for irun in range(niter):
    #         np.random.seed(irun)
    #         # Here the sigmaS is not the per-component sigma.
    #         # While, we use the std for g1+ig2, which should be sqrt(2) times of
    #         # the per component std.
    #         # Since in the transpose operation, we only keep the real part of
    #         # alpha field, such operation makes the alpha's std to 1/sqrt(2) of
    #         # the ture one. Therefore, the final estimation is correct.
    #         g1Sim   =   np.random.randn(self.nz,self.ny,self.nx)*self.sigmaS
    #         g2Sim   =   np.random.randn(self.nz,self.ny,self.nx)*self.sigmaS
    #         shearSim=   (g1Sim+np.complex64(1j)*g2Sim)*self.sigmaSInv**2.
    #         alphaRSim=  -self.chi2_transpose(shearSim) # Real field
    #         outData +=  alphaRSim**2.

    #     # masked region is assigned with the maximum of the std in this frame
    #     # and lens redshift plane
    #     maskL =   np.all(self.maskS,axis=0)
    #     for izlp in range(self.nlp):
    #         for iframe in range(self.nframe):
    #             outData[izlp,iframe][~maskL]=np.max(outData[izlp,iframe])
    #     # Calculate noise std
    #     self.sigmaA =   np.sqrt(outData/niter)
    #     self.maskA  =   np.ones(self.shapeA)

    #     # Mask the parameter field close to the boundary of the survey set the
    #     # stds of these regions to +infty
    #     for izl in range(self.nlp):
    #         for iframe in range(self.nframe):
    #             thres=np.max(self.diagonal[izl,iframe].flatten())/5.
    #             maskLP= self.diagonal[izl,iframe]>thres
    #             self.sigmaA[izl,iframe][~maskLP]=1e15
    #             self.maskA[izl,iframe][~maskLP]=0.
    #     return
