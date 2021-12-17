import json
import os
import cosmology
import numpy as np
import astropy.io.fits as pyfits
import ipyvolume as ipv
import ipyvolume.pylab as pvlt

try:
    # use lsst.afw.display and firefly backend for display
    import lsst.log as logging
    import lsst.afw.image as afwImage
    haslsst=True
except ImportError:
    haslsst=False
from configparser import ConfigParser

def zMeanBin(zMin,dz,nz):
    return np.arange(zMin,zMin+dz*nz,dz)+dz/2.

# LASSO Threshold Functions
def soft_thresholding(dum,thresholds):
    # Standard Threshold Function
    return np.sign(dum)*np.maximum(np.abs(dum)-thresholds,0.)

def firm_thresholding(dum,thresholds):
    # Glimpse13 uses f
    mask    =   (np.abs(dum)<= thresholds)
    dum[mask]=  0.
    mask    =   (np.abs(dum)>thresholds)
    mask    =   mask&(np.abs(dum)<= 2*thresholds)
    dum[mask]=  np.sign(dum[mask])*(2*np.abs(dum[mask])-thresholds[mask])
    return dum

def my_thresholding(dum,thresholds):
    # doesnot work @_@
    mask    =   (abs(dum)<= thresholds)
    dum[mask]=  0.
    mask    =   (abs(dum)>thresholds)
    dum[mask]=  np.sign(dum[mask])*(abs(dum[mask])-thresholds[mask]**2./abs(dum[mask]))
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
        if parser.has_option('sparse','eta'):           #   For ridge regulation
            self.eta   =   parser.getfloat('sparse','eta')
        else:
            self.eta   =   0.
        self.nframe =   parser.getint('sparse','nframe')
        outFname    =   'deltaMap_lbd%.1f_%s.fits' %(self.lbd,self.fieldN)
        self.outFname   =   os.path.join(self.outDir,outFname)
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
            self.zlMin       =   parser.getfloat('lensZ','zlMin')
            self.zlscale     =   parser.getfloat('lensZ','zlscale')
        self.zlBin       =   zMeanBin(self.zlMin,self.zlscale,self.nlp)

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

        self.cosmo  =   cosmology.Cosmo(h=1)
        self.lensing_kernel(self.zlBin,self.zsBin)

        dicname =   parser.get('sparse','dicname')
        if dicname=='starlet':
            from halolet import starlet2D
            self.dict2D =   starlet2D(gen=2,nframe=self.nframe,ny=self.ny,nx=self.nx)
        elif dicname=='nfwlet':
            smooth_scale =   parser.getfloat('transPlane','smooth_scale')
            from halolet import nfwlet2D
            self.dict2D =   nfwlet2D(nframe=self.nframe,ngrid=self.nx,smooth_scale=smooth_scale)
        self.ks2D   =   massmap_ks2D(self.ny,self.nx)

        # Read pixelized shear and mask
        sigfname    =   parser.get('prepare','sigmafname')
        g1fname     =   parser.get('prepare','g1fname')
        g2fname     =   parser.get('prepare','g2fname')
        self.sigmaS =   pyfits.getdata(sigfname)
        g1Map=pyfits.getdata(g1fname)
        g2Map=pyfits.getdata(g2fname)
        assert g1Map.shape  ==   self.shapeS, \
            'load wrong pixelized shear, shape map shape: (%d,%d,%d)' %g1Map.shape
        assert self.sigmaS.shape  ==   self.shapeS, \
            'load wrong pixelized std, shape map shape: (%d,%d,%d)' %self.sigmaS.shape
        self.mask   =   (self.sigmaS>=1.e-4)

        # Estimate Spectrum
        self.spectrum_norm()

        # Estimate variance plane for alpha
        self.prox_sigmaA(100)
        sigmaName   =   os.path.join(self.outDir,'sigmaA_%s.fits' %self.fieldN)
        pyfits.writeto(sigmaName,self.sigmaA[:,self.display_iframe],overwrite=True)
        muName   =   os.path.join(self.outDir,'mu_%s.fits' %self.fieldN)
        pyfits.writeto(muName,self.mu[:,self.display_iframe],overwrite=True)

        # Initialization
        self.alphaR =   np.zeros(self.shapeA)   # alpha
        self.deltaR =   np.zeros(self.shapeL)   # delta
        self.shearRRes   = np.zeros(self.shapeS)# shear residuals
        self.shearR =   g1Map+np.complex128(1j)*g2Map # shear
        return

    def lensing_kernel(self,zlbin,zsbin):
        """
        # Estimate the lensing kernel (with out poz)
        # the output is in shape (nzs,nzl)
        """
        logging.info('Calculating lensing kernel')
        if self.nlp<=1:
            self.lensKernel =   np.ones((self.nz,self.nlp))
        else:
            self.lensKernel =   np.zeros((self.nz,self.nlp))
            for i,zs in enumerate(zsbin):
                self.lensKernel[i,:]    =   self.cosmo.deltacritinv(zlbin,zs)*self.zlscale
        return

    def main_forward(self,alphaRIn):
        shearOut    =   np.zeros(self.shapeS,dtype=np.complex128)
        for zl in range(self.nlp):
            deltaFZl=   self.dict2D.itransform(alphaRIn[zl,:,:,:],inFou=False)
            shearRZl=   self.ks2D.transform(deltaFZl,outFou=False)
            shearOut+=  (self.lensKernel[:,zl,None,None]*shearRZl)
        shearOut    =   shearOut*(self.mask.astype(np.int))
        return shearOut

    def chi2_transpose(self,shearRIn):
        # initializate an empty delta map
        deltaFTmp       =   np.zeros(self.shapeL,dtype=np.complex128)
        for zs in range(self.nz):
            # For each source plane, we use KS method to get kappa
            # D operator in the paper
            kappaFZs    =   self.ks2D.itransform(shearRIn[zs],inFou=False)
            # Lensing Kernel transpose
            # to density contrast frame
            deltaFTmp  +=   (self.lensKernel[zs,:,None,None]*kappaFZs)
        # initialize a projector space
        alphaRO         =   np.empty(self.shapeA)
        for zl in range(self.nlp):
            # transpose of projection operator
            # Phi in the paper
            alphaRO[zl,:,:,:]=self.dict2D.itranspose(deltaFTmp[zl],outFou=False).real
        return alphaRO

    def gradient_chi2(self):
        # sparseBase.massmapSparsityTask.gradient_chi2
        # calculate the gradient of the chi2 component
        shearRTmp   =   self.main_forward(self.alphaR)  #A_{ij} x_j
        self.shearRRes   =   self.shearR-shearRTmp      #y_i-A_{ij} x_j
        return -self.chi2_transpose(self.shearRRes)     #-A_{i\alpha}(y_i-A_{ij} x_j)/2

    def gradient_TSV(self):
        # sparseBase.massmapSparsityTask.gradient_TSV
        # calculate the gradient of the Total Square Variance(TSV) component

        # finite difference operator
        difx    =   np.roll(self.alphaR,1,axis=-1)
        difx    =   difx-self.alphaR #D1
        # Transpose of the finite difference operator
        gradx   =   np.roll(difx,-1,axis=-1)
        gradx   =   gradx-difx # (S1)_{i\alpha} (S1)_{i\alpha} x_\alpha

        # The same for the theta2 direction
        dify    =   np.roll(self.alphaR,1,axis=-2)
        dify    =   dify-self.alphaR #D2
        grady   =   np.roll(dify,-1,axis=-2)
        grady   =   grady-dify # (S)_{ij} (S)_{i\alpha} x_j
        return (gradx+grady)*self.tau

    def gradient_ridge(self):
        # sparseBase.massmapSparsityTask.gradient_ridge
        return self.alphaR*self.eta

    def gradient(self):
        # sparseBase.massmapSparsityTask.gradient
        # calculate the gradient of the Second order component in loss function
        # wihch includes Total Square Variance(TSV) and chi2 components
        gCh2    =   self.gradient_chi2()
        gTSV    =   self.gradient_TSV()
        gRidge  =   self.gradient_ridge()
        return gCh2+gTSV+gRidge

    def spectrum_norm(self):
        # sparseBase.massmapSparsityTask.spectrum_norm
        # Estimate A_{i\alpha} A_{i\alpha}
        asquareframe=np.zeros((self.nz,self.nframe,self.ny,self.nx))
        for iz in range(self.nz):
            maskF=np.fft.fft2(self.mask[iz,:,:])
            for iframe in range(self.nframe):
                fun=np.abs(self.ks2D.transform(self.dict2D.fouaframes[iframe,:,:],outFou=False))**2.
                asquareframe[iz,iframe,:,:]=np.fft.ifft2(np.fft.fft2(fun).real*maskF).real

        spectrum=np.sum(self.lensKernel[:,:,None,None,None]**2.*asquareframe[:,None,:,:,:],axis=0)

        # the first frame
        #spectrum[:,0,:,:]=spectrum[:,0,:,:]+4.*self.tau
        self.mu0=   1./spectrum
        spectrum=   spectrum+4.*self.tau+self.eta
        self.mu =   1./spectrum
        return

    def prox_sigmaA(self,niter):
        logging.info('Estimating sigma map')
        outData     =   np.zeros(self.shapeA)
        for irun in range(niter):
            np.random.seed(irun)
            g1Sim   =   np.random.randn(self.nz,self.ny,self.nx)*self.sigmaS
            g2Sim   =   np.random.randn(self.nz,self.ny,self.nx)*self.sigmaS
            shearSim=   g1Sim+np.complex128(1j)*g2Sim
            alphaRSim=  -self.chi2_transpose(shearSim)
            outData +=  alphaRSim**2.

        #
        maskL =   np.all(self.sigmaS>1.e-4,axis=0)
        for izlp in range(self.nlp):
            for iframe in range(self.nframe):
                outData[izlp,iframe][~maskL]=np.max(outData[izlp,iframe])
        # noi_std/(A_i\alphaA_i\alpha)
        self.sigmaA =   self.mu*np.sqrt(outData/niter)
        return

    def reconstruct(self):
        #update deltaR
        for zl in range(self.nlp):
            alphaRZ         =   self.alphaR[zl].copy()
            self.deltaR[zl] =   self.dict2D.itransform(alphaRZ,inFou=False,outFou=False)
        return

    def pathwise_coordinate_descent(self,iup,niter,threM='ST'):
        self.lbd_path=np.ones(niter)*100
        ind1list    =   []
        for irun in range(1,niter):
            self.dalphaR =   -self.mu*self.gradient().real
            # Update thresholds
            snrArray=   np.abs(self.dalphaR)/self.sigmaA
            ind1d   =   np.argpartition(snrArray,-2,axis=None)[-2:]
            ind1st  =   np.unravel_index(ind1d[1],snrArray.shape)
            ind2st  =   np.unravel_index(ind1d[0],snrArray.shape)
            snr12   =   (snrArray[ind1st]+max(snrArray[ind2st],self.lbd))/2.

            self.lbd_path[irun]=min(snr12,self.lbd_path[irun-1]*0.99)
            if irun %3==1:
                print(irun,snr12,ind1st,self.lbd_path[irun],self.lbd*1.01)
            if self.lbd_path[irun]<=self.lbd*1.01:
                self.lbd_path[irun:]=self.lbd
                return
            self.thresholds=self.lbd_path[irun]*self.sigmaA

            # Determine the coordinates to update
            if ind1st not in ind1list:
                ind1list.append(ind1st)

            for i,ind in enumerate(ind1list):
                dum     =   self.alphaR+self.dalphaR
                if threM=='ST':
                    self.alphaR[ind]  = soft_thresholding(dum,self.thresholds)[ind]
                elif threM=='FT':
                    self.alphaR[ind]  = firm_thresholding(dum,self.thresholds)[ind]
                if i <len(ind1list):
                    self.dalphaR =   -self.mu*self.gradient().real

            # Write (Display) the fits file
            if irun in self.debugList:
                self.reconstruct()
                ratioPlot=self.dalphaR/self.sigmaA
                """
                for il in range(self.nlp):
                    print('max snr in %d lens bin is %.3f' %(il,ratioPlot[il].max()))
                """
                print(np.average(np.abs(self.shearRRes)**2.))
                ratioPlot[ratioPlot<self.lbd]=0.
                self.debugRatios.append(ratioPlot)
                self.debugDeltas.append(self.deltaR.real)
                self.debugAlphas.append(self.alphaR.real)
        return

    def fista_gradient_descent(self,iup,niter,threM='ST'):
        covInv=0.05 # TODO: Estimate covInv numerically?
        self.sigmaA=self.sigmaA*covInv
        self.mu=self.mu*covInv
        self.thresholds =   self.lbd*self.sigmaA
        tn=0
        for irun in range(niter):
            #save old kappaFou
            self.dalphaR =   -self.mu*self.gradient().real
            dum     =   self.alphaR+self.dalphaR
            if threM=='ST':
                dum  = soft_thresholding(dum,self.thresholds)
            elif threM=='FT':
                dum  = firm_thresholding(dum,self.thresholds)
            #update x_\alpha according to FISTA
            tnTmp= (1.+np.sqrt(1.+4*tn**2.))/2.
            ratio= (tn-1.)/tnTmp
            tn   = tnTmp
            self.alphaR=dum+(ratio*(dum-self.alphaR))
            # Write (Display) the fits file
            if irun in self.debugList:
                self.reconstruct()
        return

    def process(self,niter=1000):
        if self.aprox_method=='pathwise':
            self.pathwise_coordinate_descent(0,niter,'ST')
        elif self.aprox_method=='fista':
            self.fista_gradient_descent(0,niter,'ST')
        self.reconstruct()
        return

    def write(self):
        pyfits.writeto(self.outFname,self.deltaR.real,overwrite=True)
        return
