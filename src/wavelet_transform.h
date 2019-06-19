/*! Copyright CEA, 2015-2016
 * author : Francois Lanusse < francois.lanusse@gmail.com >
 *
 * This software is a computer program whose purpose is to reconstruct mass maps
 * from weak gravitational lensing.
 *
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 *
 */

#ifndef WAVELET_TRANSFORM_H
#define WAVELET_TRANSFORM_H

#include <fftw3.h>

#ifdef CUDA_ACC
#include <cufftXt.h>
#include <cuda_runtime.h>
#include "gpu_utils.h"
#endif

class wavelet_transform
{
    int npix, nscale, nframes, nlp;

    float **frames;
    fftwf_complex *fft_frame;

#ifdef CUDA_ACC
    cudaLibXtDesc *d_frameXt;
    cufftComplex *d_frame;
    cufftHandle  fft_plan;
    int nGPU;
    int whichGPUs[MAX_GPUS];
    size_t worksize[MAX_GPUS];
#else
    fftwf_plan plan_forward, plan_backward;
#endif


public:
    /*! Initialise Fourier based wavelet transform
     *
     */
    wavelet_transform ( int npix, int nscale, int nlp=1 );

    /*! Destructor
     *
     */
    ~wavelet_transform();

    /*! Returns the number of wavelet frames
     *
     */
    int get_nframes() {
        return nframes;
    };
    
    /*! Returns the array with wavelet transform coefficients
     * 
     */
    float ** get_frames() {
         return frames;
    }

    /*! Performs forward wavelet transform.
     *
     */
    void transform ( fftwf_complex *image, float *alpha );

    /*! Performs adjoint wavelet transform.
     *
     */
    void trans_adjoint ( float *alpha, fftwf_complex *image );
};

#endif // WAVELET_TRANSFORM_H
