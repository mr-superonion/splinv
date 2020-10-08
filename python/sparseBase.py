import os
import cosmology
import numpy as np

import json
from configparser import ConfigParser
import astropy.io.fits as pyfits


def zMeanBin(zMin,dz,nz):
    return np.arange(zMin,zMin+dz*nz,dz)+dz/2.

def soft_thresholding(dum,thresholds):
    """
    Standard Threshold Function
    """
    return np.sign(dum)*np.maximum(np.abs(dum)-thresholds,0.)

def firm_thresholding(dum,thresholds):
    """
    The firm thresholding used by Glimpse3D2013
    """
    mask    =   (np.abs(dum)<= thresholds)
    dum[mask]=  0.
    mask    =   (np.abs(dum)>thresholds)
    mask    =   mask&(np.abs(dum)<= 2*thresholds)
    dum[mask]=  np.sign(dum[mask])*(2*np.abs(dum[mask])-thresholds[mask])
    return dum

class massmap_ks2D():
    def __init__(self,ny,nx):
        self.shape   =   (ny,nx)
        self.e2phiF  =   self.e2phiFou(self.shape)

    def e2phiFou(self,shape):
        ny1,nx1 =   shape
        e2phiF  =   np.zeros(shape,dtype=complex)
        for j in range(ny1):
            jy  =   (j+ny1//2)%ny1-ny1//2
            jy  =   jy/ny1
            for i in range(nx1):
                ix  =   (i+nx1//2)%nx1-nx1//2
                ix  =   ix/nx1
                if (i**2+j**2)>0:
                    e2phiF[j,i]    =   np.complex((ix**2.-jy**2.),2.*ix*jy)/(ix**2.+jy**2.)
                else:
                    e2phiF[j,i]    =   1.
        return e2phiF*np.pi

    def itransform(self,gMap,inFou=True,outFou=True):
        assert gMap.shape==self.shape
        if not inFou:
            gMap =   np.fft.fft2(gMap)
        kOMap    =   gMap/self.e2phiF*np.pi
        if not outFou:
            kOMap    =   np.fft.ifft2(kOMap)
        return kOMap

    def transform(self,kMap,inFou=True,outFou=True):
        assert kMap.shape==self.shape
        if not inFou:
            kMap =   np.fft.fft2(kMap)
        gOMap    =   kMap*self.e2phiF/np.pi
        if not outFou:
            gOMap    =   np.fft.ifft2(gOMap)
        return gOMap

class massmapSparsityTaskNew():
    def __init__(self,parser):
        #display
        self.display_iframe=0
        if parser.has_option('file','fieldN'):
            self.fieldN =   parser.get('file','fieldN')
        else:
            self.fieldN =   'test'
        if parser.has_option('file','outDir'):
            self.outDir =   parser.get('file','outDir')
        else:
            self.outDir =   './'

        #sparse
        self.aprox_method    =   parser.get('sparse','aprox_method')
        self.lbd    =   parser.getfloat('sparse','lbd') #   For Sparsity
        if parser.has_option('sparse','tau'):           #   For Total Square Variance
            self.tau   =   parser.getfloat('sparse','tau')
        else:
            self.tau   =   0.
        self.nframe     =   parser.getint('sparse','nframe')
        minframe   =   parser.getint('sparse','minframe')
        # Do debug?
        if parser.has_option('sparse','debugList'):
            self.debugList  =   np.array(json.loads(parser.get('sparse','debugList')))
            self.debugRatios=[]
            self.debugDeltas=[]
            self.debugAlphas=[]
        else:
            debugList=[]

        ##transverse plane
        self.ny     =   parser.getint('transPlane','ny')
        self.nx     =   parser.getint('transPlane','nx')
        ##lens z axis
        self.nlp    =   parser.getint('lensZ','nlp')

        if self.nlp<=1:
            self.zlMin  =   0.
            self.zlscale=   1.
        else:
            self.zlMin  =   parser.getfloat('lensZ','zlMin')
            self.zlscale=   parser.getfloat('lensZ','zlscale')
        self.zlBin      =   zMeanBin(self.zlMin,self.zlscale,self.nlp)

        ##source z axis
        self.nz     =   parser.getint('sourceZ','nz')
        if self.nz<=1:
            self.zMin   =   0.05
            self.zscale =   2.5
            self.zsBin      =   zMeanBin(self.zMin,self.zscale,self.nz)
        else:
            zbound =   np.array(json.loads(parser.get('sourceZ','zbound')))
            self.zsBin =   (zbound[:-1]+zbound[1:])/2.

        self.shapeS =   (self.nz,self.ny,self.nx)
        self.shapeL =   (self.nlp,self.ny,self.nx)
        self.shapeA =   (self.nlp,self.nframe,self.ny,self.nx)

        # For distance calculation
        if parser.has_option('cosmology','omega_m'):
            omega_m=parser.getfloat('cosmology','omega_m')
        else:
            omega_m=0.3
        self.cosmo  =   cosmology.Cosmo(h=1,omega_m=omega_m)
        self.read_pixel_result(parser)
        #self.lensing_kernel(self.zlBin,self.zsBin)

        dicname =   parser.get('sparse','dicname')
        from halolet import nfwShearlet2D
        self.dict2D =   nfwShearlet2D(parser)
        self.ks2D   =   massmap_ks2D(self.ny,self.nx)

        # Read pixelized sigma map for shear
        sigfname    =   parser.get('prepare','sigmafname')
        self.sigmaS =   pyfits.getdata(sigfname)
        assert self.sigmaS.shape  ==   self.shapeS, \
            'load wrong pixelized std, shape map shape: (%d,%d,%d)' %self.sigmaS.shape
        self.mask   =   (self.sigmaS>=1.e-4)
        self.sigmaSInv=  np.zeros(self.shapeS)
        self.sigmaSInv[self.mask]=  1./self.sigmaS[self.mask]

        # Estimate diagonal elements of the chi2 operator
        self.fast_chi2diagonal_est()

        # Also the scale of projectors (to boost the speed)
        if self.aprox_method != 'pathwise':
            self._w =   1./np.sqrt(self.diagonal+4.*self.tau+1.e-12)
        else:
            self._w =   1.

        # Effective tau
        muRatio=   np.zeros(self.shapeA)
        for izl in range(self.nlp):
            muRatio[izl]=np.max(self.diagonal[izl].flatten())
        self.tau    =  muRatio*self.tau

        # Estimate sigma map for alpha
        self.prox_sigmaA()

        # Step size
        self.mu     =   None
        self.clear_all_outcome()

        # preparation for output
        outFname    =   'deltaMap_lbd%.1f_%s.fits' %(self.lbd,self.fieldN)
        self.outFname   =   os.path.join(self.outDir,outFname)
        return

    def clear_all_outcome(self):
        # Clear results
        self.alphaR =   np.zeros(self.shapeA)   # alpha
        self.deltaR =   np.zeros(self.shapeL)   # delta
        self.shearRRes   = np.zeros(self.shapeS)# shear residuals
        return

    def read_pixel_result(self,parser):
        g1fname     =   parser.get('prepare','g1fname')
        g2fname     =   parser.get('prepare','g2fname')
        lkfname     =   parser.get('prepare','lkfname')
        g1Map       =   pyfits.getdata(g1fname)
        g2Map       =   pyfits.getdata(g2fname)
        assert g1Map.shape  ==   self.shapeS, \
            'load wrong pixelized shear, shape should be: (%d,%d,%d)' %self.shapeS
        self.shearR =   g1Map+np.complex128(1j)*g2Map # shear

        self.lensKernel=pyfits.getdata(lkfname)
        assert self.lensKernel.shape  ==   (self.nz,self.nlp), \
            'load wrong lensing kernel, shape should be: (%d,%d)' %(self.nz,self.nlp)
        return

    """
    def lensing_kernel(self,zlBin,zsbin):
        # Old code, moved to pixel3D.py
        # Estimate the lensing kernel (with out poz)
        # the output is in shape (nzs,nzl)
        if self.nlp<=1:
            self.lensKernel =   np.ones((self.nz,self.nlp))
        else:
            self.lensKernel =   np.zeros((self.nz,self.nlp))
            for i,zs in enumerate(zsbin):
                self.lensKernel[i,:]    =   self.cosmo.deltacritinv(zlBin,zs)*self.zlscale
        return
    """

    def get_basis_vector(self,ind):
        assert np.all((np.array(self.shapeA)-np.array(ind))>=0),\
            'index not in the projector space'
        alphaTmp    =   np.zeros(self.shapeA)
        alphaTmp[ind]=  1.
        return self.main_forward(alphaTmp)

    def main_forward(self,alphaRIn):
        """
        Transform from dictionary space to observational space

        Parameters
        ----------
        alphaRIn: modes in dictionary space.

        """
        # self._w is the weight on the forward operator
        alphaRIn    =   alphaRIn*self._w
        shearOut    =   self.dict2D.itransform(alphaRIn)
        shearOut    =   shearOut*(self.mask.astype(np.int))
        return shearOut

    def chi2_transpose(self,shearRIn):
        """
        Traspose operation on observed map

        Parameters
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

        Parameters
        ----------
        alphaRIn: modes in dictionary space.

        """
        # sparseBase.massmapSparsityTask.gradient_chi2
        # calculate the gradient of the chi2 component
        shearRTmp       =   self.main_forward(alphaR)               #A_{ij} x_j
        self.shearRRes  =   (self.shearR-shearRTmp)*self.sigmaSInv  #y_i-A_{ij} x_j
        return -self.chi2_transpose(self.shearRRes*self.sigmaSInv)  #-A_{i\alpha}(y_i-A_{ij}x_j)

    def gradient_TSV(self,alphaR):
        # calculate the gradient of the Total Square Variance(TSV) component
        # finite difference operator
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

        # for the z direction
        """
        difz    =   np.roll(alphaR,1,axis=-2)
        difz    =   difz+alphaR #D2
        gradz   =   np.roll(difz,-1,axis=-2)
        gradz   =   gradz+difz # (S)_{ij} (S_2)_{i\alpha} x_\alpha
        """
        gradz   =   0.
        return (gradx+grady+gradz)*self.tau*self._w

    def quad_gradient(self,alphaR):
        # sparseBase.massmapSparsityTask.gradient
        # calculate the gradient of the Second order component in loss function
        # wihch includes Total Square Variance(TSV) and chi2 components
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
        asquareframe=   np.zeros((self.nz,self.nframe,self.ny,self.nx))
        for iz in range(self.nz):
            maskF   =   np.fft.fft2((self.sigmaSInv[iz]**2.))
            for iframe in range(self.nframe):
                fun=np.conj(self.dict2D.aframes[iz,iframe])*self.dict2D.aframes[iz,iframe]
                asquareframe[iz,iframe,:,:]=np.fft.ifft2(np.fft.fft2(fun)*maskF).real

        self.diagonal=  np.sum(self.lensKernel[:,:,None,None,None]**2.\
                *asquareframe[:,None,:,:,:],axis=0)
        return

    def determine_step_size(self):
        norm        =   0.
        for irun in range(1000):
            # generate a normalized random vector
            np.random.seed(irun)
            alphaTmp=   np.random.randn(self.nlp,self.nframe,self.ny,self.nx)
            normTmp =   np.sqrt(np.sum(alphaTmp**2.))
            alphaTmp=   alphaTmp/normTmp
            # apply the transform matrix to the vector
            alphaTmp2=  self.quad_gradient(alphaTmp)
            normTmp2=   np.sqrt(np.sum(alphaTmp2**2.))
            if normTmp2>norm:
                norm=normTmp2
        self.mu    = 1./norm/1.8
        return

    def prox_sigmaA(self):
        """Calculate stds of the paramters
        Note that the std should be all 1 since we normalize the
        projectors.
        However, we set some stds to +infty
        """
        niter   =   100
        # A_i\alpha n_i
        outData     =   np.zeros(self.shapeA)
        for irun in range(niter):
            np.random.seed(irun)
            g1Sim   =   np.random.randn(self.nz,self.ny,self.nx)*self.sigmaS
            g2Sim   =   np.random.randn(self.nz,self.ny,self.nx)*self.sigmaS
            shearSim=   (g1Sim+np.complex128(1j)*g2Sim)*self.sigmaSInv**2.
            alphaRSim=  -self.chi2_transpose(shearSim)
            outData +=  alphaRSim**2.

        #TODO: Dowe really need it?
        # masked region is assigned with the maximum
        maskL =   np.all(self.sigmaS>1.e-4,axis=0)
        for izlp in range(self.nlp):
            for iframe in range(self.nframe):
                outData[izlp,iframe][~maskL]=np.max(outData[izlp,iframe])
        # noi std
        self.sigmaA =   np.sqrt(outData/niter)

        # The structures outside of the boundary
        # set the stds to +infty
        for izl in range(self.nlp):
            for iframe in range(self.nframe):
                thres=np.max(self.diagonal[izl,iframe].flatten())/10.
                maskLP= self.diagonal[izl,iframe]>thres
                self.sigmaA[izl,iframe][~maskLP]=1e12
        return

    def reconstruct(self):
        #update deltaR
        alphaRW         =   self.alphaR.copy()*self._w
        # Only keep the E-mode
        self.deltaR     =   self.dict2D.itransformDelta(alphaRW).real
        return

    def adaptive_lasso_weight(self,gamma=1,sm_scale=0.25):
        """Calculate adaptive weight
        @param gamma:       power of the root-n consistent (preliminary)
                            estimation
        @param sm_scale:    top-hat smoothing scale for the root-n
                            consistent estimation [Mpc/h]
        """
        # Smoothing scale in arcmin
        rsmth0=np.zeros(self.nlp,dtype=int)
        for iz,zh in enumerate(self.zlBin):
            rsmth0[iz]=(np.round(sm_scale/self.cosmo.Dc(0.,zh)*60*180./np.pi))

        p   =   np.zeros(self.shapeA)
        if self.nframe==1 and sm_scale>1e-4:
            for izl in range(self.nlp):
                rsmth   =   rsmth0[izl]
                for jsh in range(-rsmth,rsmth+1):
                    # only smooth the point mass frame
                    dif    =   np.roll(self.alphaR[izl,0],jsh,axis=-2)
                    for ish in range(-rsmth,rsmth+1):
                        dif2=  np.roll(dif,ish,axis=-1)
                        p[izl,0] += dif2/(2.*rsmth+1.)#**2.
            p   =   np.abs(p)
        else:
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
        if self.mu is None:
            # Determine step size self.mu
            self.determine_step_size()
        # The thresholds
        thresholds =   self.lbd*self.sigmaA*self.mu*w
        # FISTA algorithms
        tn      =   1.
        Xp0     =   self.alphaR
        for irun in range(niter):
            dalphaR =   -self.mu*self.quad_gradient(self.alphaR).real
            Xp1 =   self.alphaR+dalphaR
            Xp1 =   soft_thresholding(Xp1,thresholds)
            tnTmp= (1.+np.sqrt(1.+4*tn**2.))/2.
            ratio= (tn-1.)/tnTmp
            self.alphaR=Xp1+(ratio*(Xp1-Xp0))
            tn  = tnTmp
            Xp0 =   Xp1
            if irun in self.debugList:
                self.reconstruct()
        return

    def process(self,niter=1000):
        self.fista_gradient_descent(niter)
        self.reconstruct()
        return

    def write(self):
        pyfits.writeto(self.outFname,self.deltaR.real,overwrite=True)
        return

class massmapSparsityTask():
    def __init__(self,parser):
        #display
        self.display_iframe=0
        if parser.has_option('file','fieldN'):
            self.fieldN =   parser.get('file','fieldN')
        else:
            self.fieldN =   'test'
        if parser.has_option('file','outDir'):
            self.outDir =   parser.get('file','outDir')
        else:
            self.outDir =   './'

        #sparse
        self.aprox_method    =   parser.get('sparse','aprox_method')
        self.lbd    =   parser.getfloat('sparse','lbd') #   For Sparsity
        if parser.has_option('sparse','tau'):           #   For Total Square Variance
            self.tau   =   parser.getfloat('sparse','tau')
        else:
            self.tau   =   0.
        self.nframe     =   parser.getint('sparse','nframe')
        minframe   =   parser.getint('sparse','minframe')
        # Do debug?
        if parser.has_option('sparse','debugList'):
            self.debugList  =   np.array(json.loads(parser.get('sparse','debugList')))
            self.debugRatios=[]
            self.debugDeltas=[]
            self.debugAlphas=[]
        else:
            debugList=[]

        ##transverse plane
        self.ny     =   parser.getint('transPlane','ny')
        self.nx     =   parser.getint('transPlane','nx')
        ##lens z axis
        self.nlp    =   parser.getint('lensZ','nlp')

        if self.nlp<=1:
            self.zlMin  =   0.
            self.zlscale=   1.
        else:
            self.zlMin  =   parser.getfloat('lensZ','zlMin')
            self.zlscale=   parser.getfloat('lensZ','zlscale')
        self.zlBin      =   zMeanBin(self.zlMin,self.zlscale,self.nlp)

        ##source z axis
        self.nz     =   parser.getint('sourceZ','nz')
        if self.nz<=1:
            self.zMin   =   0.05
            self.zscale =   2.5
            self.zsBin      =   zMeanBin(self.zMin,self.zscale,self.nz)
        else:
            zbound =   np.array(json.loads(parser.get('sourceZ','zbound')))
            self.zsBin =   (zbound[:-1]+zbound[1:])/2.

        self.shapeS =   (self.nz,self.ny,self.nx)
        self.shapeL =   (self.nlp,self.ny,self.nx)
        self.shapeA =   (self.nlp,self.nframe,self.ny,self.nx)

        # For distance calculation
        if parser.has_option('cosmology','omega_m'):
            omega_m=parser.getfloat('cosmology','omega_m')
        else:
            omega_m=0.3
        self.cosmo  =   cosmology.Cosmo(h=1,omega_m=omega_m)
        self.read_pixel_result(parser)
        #self.lensing_kernel(self.zlBin,self.zsBin)

        dicname =   parser.get('sparse','dicname')
        if dicname=='starlet':
            from halolet import starlet2D
            self.dict2D =   starlet2D(gen=2,nframe=self.nframe,ny=self.ny,nx=self.nx)
        elif dicname=='nfwlet':
            smooth_scale =   parser.getfloat('transPlane','smooth_scale')
            from halolet import nfwlet2D
            self.dict2D =   nfwlet2D(nframe=self.nframe,minframe=minframe,ngrid=self.nx,smooth_scale=smooth_scale)
        self.ks2D   =   massmap_ks2D(self.ny,self.nx)
        self.shearNolensKerAtom =   np.zeros((self.nframe,self.ny,self.nx),dtype=np.complex128)
        for iframe in range(self.nframe):
            self.shearNolensKerAtom[iframe,:,:] \
            = self.ks2D.transform(self.dict2D.fouaframes[iframe,:,:],outFou=False)

        # Read pixelized sigma map for shear
        sigfname    =   parser.get('prepare','sigmafname')
        self.sigmaS =   pyfits.getdata(sigfname)
        assert self.sigmaS.shape  ==   self.shapeS, \
            'load wrong pixelized std, shape map shape: (%d,%d,%d)' %self.sigmaS.shape
        self.mask   =   (self.sigmaS>=1.e-4)
        self.sigmaSInv=  np.zeros(self.shapeS)
        self.sigmaSInv[self.mask]=  1./self.sigmaS[self.mask]

        # Estimate diagonal elements of the chi2 operator
        self.fast_chi2diagonal_est()

        # Also the scale of projectors (to boost the speed)
        if self.aprox_method != 'pathwise':
            self._w =   1./np.sqrt(self.diagonal+4.*self.tau+1.e-12)
        else:
            self._w =   1.

        # Effective tau
        muRatio=   np.zeros(self.shapeA)
        for izl in range(self.nlp):
            muRatio[izl]=np.max(self.diagonal[izl].flatten())
        self.tau    =  muRatio*self.tau

        # Estimate sigma map for alpha
        self.prox_sigmaA()

        # Step size
        self.mu     =   None
        self.clear_all_outcome()

        # preparation for output
        outFname    =   'deltaMap_lbd%.1f_%s.fits' %(self.lbd,self.fieldN)
        self.outFname   =   os.path.join(self.outDir,outFname)
        return

    def clear_all_outcome(self):
        # Clear results
        self.alphaR =   np.zeros(self.shapeA)   # alpha
        self.deltaR =   np.zeros(self.shapeL)   # delta
        self.shearRRes   = np.zeros(self.shapeS)# shear residuals
        return

    def read_pixel_result(self,parser):
        g1fname     =   parser.get('prepare','g1fname')
        g2fname     =   parser.get('prepare','g2fname')
        lkfname     =   parser.get('prepare','lkfname')
        g1Map       =   pyfits.getdata(g1fname)
        g2Map       =   pyfits.getdata(g2fname)
        self.lensKernel=pyfits.getdata(lkfname)
        assert g1Map.shape  ==   self.shapeS, \
            'load wrong pixelized shear, shape should be: (%d,%d,%d)' %self.shapeS
        self.shearR =   g1Map+np.complex128(1j)*g2Map # shear
        assert self.lensKernel.shape  ==   (self.nz,self.nlp), \
            'load wrong lensing kernel, shape should be: (%d,%d)' %(self.nz,self.nlp)
        return

    """
    def lensing_kernel(self,zlBin,zsbin):
        # Old code, moved to pixel3D.py
        # Estimate the lensing kernel (with out poz)
        # the output is in shape (nzs,nzl)
        if self.nlp<=1:
            self.lensKernel =   np.ones((self.nz,self.nlp))
        else:
            self.lensKernel =   np.zeros((self.nz,self.nlp))
            for i,zs in enumerate(zsbin):
                self.lensKernel[i,:]    =   self.cosmo.deltacritinv(zlBin,zs)*self.zlscale
        return
    """

    def get_basis_vector(self,ind):
        assert np.all((np.array(self.shapeA)-np.array(ind))>=0),\
            'index not in the projector space'
        alphaTmp    =   np.zeros(self.shapeA)
        alphaTmp[ind]=  1.
        return self.main_forward(alphaTmp)

    def main_forward(self,alphaRIn):
        """
        Transform from dictionary space to observational space

        Parameters
        ----------
        alphaRIn: modes in dictionary space.

        """
        # self._w is the weight on the forward operator
        alphaRIn    =   alphaRIn*self._w
        shearOut    =   np.zeros(self.shapeS,dtype=np.complex128)
        for zl in range(self.nlp):
            deltaFZl=   self.dict2D.itransform(alphaRIn[zl,:,:,:],inFou=False)
            shearRZl=   self.ks2D.transform(deltaFZl,outFou=False)
            shearOut+=  (self.lensKernel[:,zl,None,None]*shearRZl)
        shearOut    =   shearOut*(self.mask.astype(np.int))
        return shearOut

    def chi2_transpose(self,shearRIn):
        """
        Traspose operation on observed map

        Parameters
        ----------
        shearRIn: input observed map (e.g. opbserved shear map)

        """

        # initializate an empty delta map
        alphaRO         =   np.zeros(self.shapeA,dtype=np.float64)
        for zs in range(self.nz):
            # For each source plane, we use KS method to get kappa
            kappaFZs    =   self.ks2D.itransform(shearRIn[zs],inFou=False)
            # Then project to alpha (model) space on source plane
            alphaRZs    =   self.dict2D.itranspose(kappaFZs,outFou=False).real
            # Lensing Kernel transpose to density contrast frame
            alphaRO     +=   (self.lensKernel[zs,:,None,None,None]*alphaRZs)
        return alphaRO*self._w

    def gradient_chi2(self,alphaR):
        """
        Gradient operation of Chi2 act on dictionary space

        Parameters
        ----------
        alphaRIn: modes in dictionary space.

        """
        shearRTmp       =   self.main_forward(alphaR)               #A_{ij} x_j
        self.shearRRes  =   (self.shearR-shearRTmp)*self.sigmaSInv  #y_i-A_{ij} x_j
        return -self.chi2_transpose(self.shearRRes*self.sigmaSInv)  #-A_{i\alpha}(y_i-A_{ij}x_j)

    def gradient_TSV(self,alphaR):
        # calculate the gradient of the Total Square Variance(TSV) component
        # finite difference operator
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

        # for the z direction
        """
        difz    =   np.roll(alphaR,1,axis=-2)
        difz    =   difz+alphaR #D2
        gradz   =   np.roll(difz,-1,axis=-2)
        gradz   =   gradz+difz # (S)_{ij} (S_2)_{i\alpha} x_\alpha
        """
        gradz   =   0.
        return (gradx+grady+gradz)*self.tau*self._w

    def quad_gradient(self,alphaR):
        # sparseBase.massmapSparsityTask.gradient
        # calculate the gradient of the Second order component in loss function
        # wihch includes Total Square Variance(TSV) and chi2 components
        gCh2    =   self.gradient_chi2(alphaR)
        if np.max(self.tau)>0.:
            gTSV    =   self.gradient_TSV(alphaR)
        else:
            gTSV    =   0.
        return gCh2+gTSV

    def get_basis_cov(self,ind1,ind2):
        shearLR1=   np.roll(np.roll(self.shearNolensKerAtom[\
            ind1[1]],ind1[2],axis=0),ind1[3],axis=1)
        shearLR2=   np.roll(np.roll(self.shearNolensKerAtom[\
            ind2[1]],ind2[2],axis=0),ind2[3],axis=1)
        shearLR1=   shearLR1[None,:,:]*self.lensKernel[:,ind1[0],None,None]
        shearLR2=   shearLR2[None,:,:]*self.lensKernel[:,ind2[0],None,None]
        return np.sum(np.conj(shearLR1)*shearLR2*self.mask).real

    def fast_chi2diagonal_est(self):
        """
        Estimate the diagonal elements of the Chi2 transform matrix
        """
        asquareframe=   np.zeros((self.nz,self.nframe,self.ny,self.nx))
        for iz in range(self.nz):
            maskF   =   np.fft.fft2((self.sigmaSInv[iz]**2.))
            for iframe in range(self.nframe):
                fun=np.abs(self.shearNolensKerAtom[iframe,:,:])**2.
                asquareframe[iz,iframe,:,:]=np.fft.ifft2(np.fft.fft2(fun).real*maskF).real

        self.diagonal=  np.sum(self.lensKernel[:,:,None,None,None]**2.\
                *asquareframe[:,None,:,:,:],axis=0)
        return

    def determine_step_size(self):
        norm        =   0.
        for irun in range(1000):
            # generate a normalized random vector
            np.random.seed(irun)
            alphaTmp=   np.random.randn(self.nlp,self.nframe,self.ny,self.nx)
            normTmp =   np.sqrt(np.sum(alphaTmp**2.))
            alphaTmp=   alphaTmp/normTmp
            # apply the transform matrix to the vector
            alphaTmp2=  self.quad_gradient(alphaTmp)
            normTmp2=   np.sqrt(np.sum(alphaTmp2**2.))
            if normTmp2>norm:
                norm=normTmp2
        self.mu    = 1./norm/2.
        return

    def prox_sigmaA(self):
        """Calculate stds of the paramters
        Note that the std should be all 1 since we normalize the
        projectors.
        However, we set some stds to +infty
        """
        niter   =   100
        # A_i\alpha n_i
        outData     =   np.zeros(self.shapeA)
        for irun in range(niter):
            np.random.seed(irun)
            g1Sim   =   np.random.randn(self.nz,self.ny,self.nx)*self.sigmaS
            g2Sim   =   np.random.randn(self.nz,self.ny,self.nx)*self.sigmaS
            shearSim=   (g1Sim+np.complex128(1j)*g2Sim)*self.sigmaSInv**2.
            alphaRSim=  -self.chi2_transpose(shearSim)
            outData +=  alphaRSim**2.

        #TODO: Dowe really need it?
        # masked region is assigned with the maximum
        maskL =   np.all(self.sigmaS>1.e-4,axis=0)
        for izlp in range(self.nlp):
            for iframe in range(self.nframe):
                outData[izlp,iframe][~maskL]=np.max(outData[izlp,iframe])
        # noi std
        self.sigmaA =   np.sqrt(outData/niter)

        # The structures outside of the boundary
        # set the stds to +infty
        for izl in range(self.nlp):
            for iframe in range(self.nframe):
                thres=np.max(self.diagonal[izl,iframe].flatten())/10.
                maskLP= self.diagonal[izl,iframe]>thres
                self.sigmaA[izl,iframe][~maskLP]=1e12
        return

    def reconstruct(self):
        #update deltaR
        alphaRW         =   self.alphaR.copy()*self._w
        for zl in range(self.nlp):
            self.deltaR[zl] =   self.dict2D.itransform(alphaRW[zl],inFou=False,outFou=False)
        mask=(np.abs(self.deltaR)>1e-3)
        self.deltaR[~mask]=0.
        return

    def pathwise_coordinate_descent(self,niter):
        # The step size is the inverse of the
        # diagonal for coordinate descent
        self.mu         =   1./(self.diagonal+4.*self.tau)
        thresholds =   self.lbd*self.sigmaA*self.mu

        # Do it pathwisely
        self.lbd_path   =   np.ones(niter)*100
        self.ind1list   =   []
        self.projShear  =   self.project_obs_shear()
        projCor         =   []
        for irun in range(1,niter):
            self.dalphaR=   -self.mu*self.quad_gradient(self.alphaR).real
            # Update thresholds
            snrArray=   np.abs(self.dalphaR)/(self.sigmaA*self.mu)
            ind1d   =   np.argpartition(snrArray,-2,axis=None)[-2:]
            ind1st  =   np.unravel_index(ind1d[1],snrArray.shape)
            ind2st  =   np.unravel_index(ind1d[0],snrArray.shape)
            snr12   =   (snrArray[ind1st]+max(snrArray[ind2st],self.lbd))/2.
            self.lbd_path[irun]=snr12
            if self.lbd_path[irun]<=self.lbd*1.01:
                self.lbd_path[irun:]=self.lbd
                return

            if irun%5==1:
                print(irun,snr12,ind1st,self.lbd_path[irun],self.lbd*1.01)

            # update the maximum projector
            dum     =   self.alphaR[ind1st]+self.dalphaR[ind1st]
            self.alphaR[ind1st]  = soft_thresholding(dum,thresholds[ind1st])

            # update the index list
            if ind1st not in self.ind1list:
                # if we find new maximum index
                # add it to our targets
                cors    =   []
                for ind in self.ind1list:
                    cors.append(self.get_basis_cov(ind,ind1st))
                cors.append(self.diagonal[ind1st])
                self.ind1list.append(ind1st)
                projCor.append(cors)
                # make a symmetix matrix
                pad =   len(max(projCor, key=len))
                atmp=   np.array([i + [0]*(pad-len(i)) for i in projCor])
                self.projCor=atmp + atmp.T - np.diag(atmp.diagonal())
                assert self.projCor.shape[0]==len(self.ind1list)

            # update projector
            for indid in range(min(len(self.ind1list),10)):
                indid   =   np.random.randint(len(self.ind1list))
                indU    =   self.ind1list[indid]
                pps     =   np.array([self.alphaR[ii] for ii in self.ind1list])
                dch2    =   self.mu[indU]*(self.projShear[indU]-np.sum(self.projCor[indid]*pps))
                dTSV    =   -self.mu[indU]*self.gradient_TSV()[indU]

                dum     =   self.alphaR[indU]+dch2+dTSV
                self.alphaR[indU]  = soft_thresholding(dum,thresholds[indU])

            # Write (Display) the fits file
            if irun in self.debugList:
                self.reconstruct()
                ratioPlot=self.dalphaR/self.sigmaA
                #for il in range(self.nlp):
                #    print('max snr in %d lens bin is %.3f' %(il,ratioPlot[il].max()))
                ratioPlot[ratioPlot<self.lbd]=0.
                self.debugRatios.append(ratioPlot)
                self.debugDeltas.append(self.deltaR.real)
                self.debugAlphas.append(self.alphaR.real)
        return

    def adaptive_lasso_weight(self,gamma=1,sm_scale=0.25):
        """Calculate adaptive weight
        @param gamma:       power of the root-n consistent (preliminary)
                            estimation
        @param sm_scale:    top-hat smoothing scale for the root-n
                            consistent estimation [Mpc/h]
        """
        # Smoothing scale in arcmin
        rsmth0=np.zeros(self.nlp,dtype=int)
        for iz,zh in enumerate(self.zlBin):
            rsmth0[iz]=(np.round(sm_scale/self.cosmo.Dc(0.,zh)*60*180./np.pi))

        p   =   np.zeros(self.shapeA)
        for izl in range(self.nlp):
            rsmth   =   rsmth0[izl]
            for jsh in range(-rsmth,rsmth+1):
                # only smooth the point mass frame
                dif    =   np.roll(self.alphaR[izl,0],jsh,axis=-2)
                for ish in range(-rsmth,rsmth+1):
                    dif2=  np.roll(dif,ish,axis=-1)
                    p[izl,0] += dif2/(2.*rsmth+1.)#**2.
        p   =   np.abs(p)

        # threshold(for value close to zero)
        thres_adp=  1./1e12
        mask=   (p**gamma>thres_adp)

        # weight estimation
        w       =   np.zeros(self.shapeA)
        w[mask] =   1./(p[mask])**(gamma)
        w[~mask]=   1./thres_adp
        return w

    def fista_gradient_descent(self,niter,w=1.):
        if self.mu is None:
            # Determine step size self.mu
            self.determine_step_size()
        # The thresholds
        thresholds =   self.lbd*self.sigmaA*self.mu*w
        # FISTA algorithms
        tn      =   1.
        Xp0     =   self.alphaR
        for irun in range(niter):
            dalphaR =   -self.mu*self.quad_gradient(self.alphaR).real
            Xp1 =   self.alphaR+dalphaR
            Xp1 =   soft_thresholding(Xp1,thresholds)
            tnTmp= (1.+np.sqrt(1.+4*tn**2.))/2.
            ratio= (tn-1.)/tnTmp
            self.alphaR=Xp1+(ratio*(Xp1-Xp0))
            tn  = tnTmp
            Xp0 =   Xp1
            if irun in self.debugList:
                self.reconstruct()
        return

    def process(self,niter=1000):
        if self.aprox_method=='pathwise':
            self.pathwise_coordinate_descent(niter)
        elif self.aprox_method=='fista':
            self.fista_gradient_descent(niter)
        self.reconstruct()
        return

    def write(self):
        pyfits.writeto(self.outFname,self.deltaR.real,overwrite=True)
        return
