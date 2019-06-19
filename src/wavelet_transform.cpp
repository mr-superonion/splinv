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

#include <sparse2d/MR_Obj.h>
#include "wavelet_transform.h"
#include "starlet_2d.h"

wavelet_transform::wavelet_transform(int npix, int nscale, int nlp):
    npix(npix), nscale(nscale), nlp(nlp)
{

    fftw_complex *frame = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * npix * npix);
    fftw_plan plan  = fftw_plan_dft_2d(npix, npix, frame, frame, FFTW_FORWARD,   FFTW_MEASURE);

    // We begin with starlets combined with battle_lemarie wavelets
    nframes = nscale + 3; // all starlet frames + the 3 first BL scales

    frames = new float*[nframes];
    for (int i = 0; i < nframes; i++) {
        frames[i] = (float *) fftwf_malloc(sizeof(float) * npix * npix);
    }

    // We extract atoms for each frame
    starlet_2d star(npix, npix, nscale);

    dblarray image(npix, npix);
    image.init(0);
    image(npix / 2, npix / 2) = 1.0;
    dblarray alphaStar(npix, npix, nscale);
    star.transform_gen1(image.buffer(), alphaStar.buffer());

    for (int i = 0; i < nscale; i++) {
        for (long ind = 0; ind < npix * npix; ind++) {
            frame[ind][0] = alphaStar.buffer()[i * npix * npix + ind];
            frame[ind][1] = 0;
        }

        fftw_execute(plan);

        for (long ind = 0; ind < npix * npix; ind++) {
            frames[i][ind] = sqrt(frame[ind][0] * frame[ind][0] + frame[ind][1] * frame[ind][1])/ npix;
        }
    }

    MultiResol mr;
    FilterAnaSynt FAS;
    FilterAnaSynt *PtrFAS = NULL;
    FAS.alloc(F_LEMARIE_5);
    PtrFAS = &FAS;
    mr.alloc(npix, npix, nscale, TO_UNDECIMATED_MALLAT, PtrFAS);

    Ifloat  im(npix, npix);
    im(npix / 2, npix / 2) = 1.0;
    mr.transform(im);
    for (int i = 0; i < 3; i++) { // For all bands replace 3 by mr.nbr_band()
        im = mr.band(i);
        for (long x = 0; x < npix; x++) {
            int k1 = x - npix / 2;
            k1 = k1 < 0 ? npix + k1 : k1;
            for (long y = 0; y < npix; y++) {
                int k2 = y - npix / 2;
                k2 = k2 < 0 ? npix + k2 : k2;
                frame[x + npix * y][0] = im.buffer()[k1 + npix * k2];
                frame[x + npix * y][1] = 0;
            }
        }

        fftw_execute(plan);

        for (long ind = 0; ind < npix * npix ; ind++) {
            frames[i + nscale][ind] = sqrt(frame[ind][0] * frame[ind][0] + frame[ind][1] * frame[ind][1])/ npix;
        }
    }

    fftw_free(frame);
    fftw_destroy_plan(plan);

    // Allocate batch wavelet transform either using fftw or CUDA
    fft_frame = fftwf_alloc_complex(npix * npix * nlp * nframes);

    int dimensions[2] = { npix, npix };
    int rank = 2;

#ifdef CUDA_ACC
    cufftResult ret = cufftCreate(&fft_plan);

    // Look for the number of available GPUs
    getDeviceCount(&nGPU);
    getGPUs(whichGPUs);
        
    std::cout << "Performing wavelet transform using " << nGPU << " GPUs" <<std::endl; 
    // 2 cases: Single GPU or Multiple GPUs
    if(nGPU > 1){

        ret =cufftXtSetGPUs(fft_plan , nGPU, whichGPUs);
        if(ret != 0) std::cout <<"set gpus" << ret << std::endl;
        
        ret =cufftMakePlanMany(fft_plan, rank, dimensions,
                        NULL, 1, npix*npix, NULL, 1, npix*npix,
                        CUFFT_C2C, nlp * nframes, worksize);
        if(ret != 0) std::cout <<"make plan " << ret << std::endl;
        
        ret = cufftXtMalloc(fft_plan, &d_frameXt, CUFFT_XT_FORMAT_INPLACE);
        if(ret != 0) std::cout <<"malloc " << ret << std::endl;
     }else{
         
        // Select GPU
        cudaSetDevice(whichGPUs[0]);

        ret =cufftMakePlanMany(fft_plan, rank, dimensions,
                        NULL, 1, npix*npix, NULL, 1, npix*npix,
                        CUFFT_C2C, nlp * nframes, worksize);
        if(ret != 0) std::cout <<"make plan " << ret << std::endl;
        
        cudaMalloc(&d_frame, sizeof(cufftComplex)*nlp*nframes*npix*npix);
     }
#else
    plan_forward = fftwf_plan_many_dft(rank, dimensions, nlp * nframes,
                                      fft_frame, NULL, 1, npix*npix,
                                      fft_frame, NULL, 1, npix*npix,
                                      FFTW_FORWARD, FFTW_MEASURE);
    plan_backward = fftwf_plan_many_dft(rank, dimensions, nlp * nframes,
                                       fft_frame, NULL, 1, npix*npix,
                                       fft_frame, NULL, 1, npix*npix,
                                       FFTW_BACKWARD, FFTW_MEASURE);
#endif
}

wavelet_transform::~wavelet_transform()
{

#ifdef CUDA_ACC
    if(nGPU>1){
    cufftXtFree(d_frameXt);
    }else{
     cudaFree(d_frame);
    }
    cufftDestroy(fft_plan);

#else
    fftwf_destroy_plan(plan_backward);
    fftwf_destroy_plan(plan_forward);

#endif

    fftwf_free(fft_frame);
    for (int i = 0; i < nframes; i++) {
        free(frames[i]);
    }
    delete[] frames;
}

void wavelet_transform::transform(fftwf_complex *image, float *alpha)
{
    #pragma omp parallel
    for (int z = 0; z < nlp; z++) {
        for (int i = 0; i < nframes; i++) {
            #pragma omp for
            for (long ind = 0; ind < npix * npix; ind++) {
                fft_frame[ind + i * npix * npix + z * npix * npix * nframes][0] = image[ind + z * npix * npix][0] * frames[i][ind];
                fft_frame[ind + i * npix * npix + z * npix * npix * nframes][1] = image[ind + z * npix * npix][1] * frames[i][ind];
            }
        }
    }

#ifdef CUDA_ACC
    if(nGPU>1){
        cufftXtMemcpy(fft_plan, d_frameXt, fft_frame, CUFFT_COPY_HOST_TO_DEVICE);
        cufftXtExecDescriptorC2C(fft_plan, d_frameXt, d_frameXt, CUFFT_INVERSE);
        cufftXtMemcpy(fft_plan, fft_frame, d_frameXt, CUFFT_COPY_DEVICE_TO_HOST);
    }else{
        cudaMemcpy(d_frame, fft_frame, sizeof(cufftComplex)* npix*npix*nlp*nframes, cudaMemcpyHostToDevice);
        cufftExecC2C(fft_plan,d_frame,d_frame, CUFFT_INVERSE);
        cudaMemcpy(fft_frame, d_frame, sizeof(cufftComplex)* npix*npix*nlp*nframes, cudaMemcpyDeviceToHost);
    }
#else
    fftwf_execute(plan_backward);
#endif

    #pragma omp parallel
    for (int z = 0; z < nlp; z++) {
        for (int i = 0; i < nframes; i++) {
            #pragma omp for
            for (long ind = 0; ind < npix * npix; ind++) {
                alpha[ind + i * npix * npix + z * npix * npix * nframes] = fft_frame[ind + i * npix * npix + z * npix * npix * nframes][0];
            }
        }
    }
}

void wavelet_transform::trans_adjoint(float *alpha, fftwf_complex *image)
{

    #pragma omp parallel for
    for (long ind = 0; ind < npix * npix * nlp; ind++) {
        image[ind][0] = 0;
        image[ind][1] = 0;
    }

    #pragma omp parallel
    for (int z = 0; z < nlp; z++) {
        for (int i = 0; i < nframes; i++) {

            #pragma omp for
            for (long ind = 0; ind < npix * npix; ind++) {
                fft_frame[ind + i * npix * npix + z * npix * npix * nframes][0] = alpha[ind + i * npix * npix + z * npix * npix * nframes];
                fft_frame[ind + i * npix * npix + z * npix * npix * nframes][1] = 0;
            }
        }
    }

#ifdef CUDA_ACC
    if(nGPU>1){
        cufftXtMemcpy(fft_plan, d_frameXt, fft_frame, CUFFT_COPY_HOST_TO_DEVICE);
        cufftXtExecDescriptorC2C(fft_plan, d_frameXt, d_frameXt, CUFFT_FORWARD);
        cufftXtMemcpy(fft_plan, fft_frame, d_frameXt, CUFFT_COPY_DEVICE_TO_HOST);
    }else{
        cudaMemcpy(d_frame, fft_frame, sizeof(cufftComplex)* npix*npix*nlp*nframes, cudaMemcpyHostToDevice);
        cufftExecC2C(fft_plan,d_frame,d_frame, CUFFT_FORWARD);
        cudaMemcpy(fft_frame, d_frame, sizeof(cufftComplex)* npix*npix*nlp*nframes, cudaMemcpyDeviceToHost);
    }
#else
    fftwf_execute(plan_forward);
#endif

    #pragma omp parallel
    for (int z = 0; z < nlp; z++) {
        for (int i = 0; i < nframes; i++) {

            #pragma omp for
            for (long ind = 0; ind < npix * npix; ind++) {
                image[ind + z * npix * npix][0] += fft_frame[ind + i * npix * npix + z * npix * npix * nframes][0] * frames[i][ind];
                image[ind + z * npix * npix][1] += fft_frame[ind + i * npix * npix + z * npix * npix * nframes][1] * frames[i][ind];
            }
        }
    }
}
