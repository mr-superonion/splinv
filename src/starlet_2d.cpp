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

#include "starlet_2d.h"
#include <math.h>
#include <iostream>

#define test_index_zero(ind , Nind, tab, offset ) ( (ind) >= Nind ? 0 : ( (ind) < 0 ? 0 : tab[ offset (ind) ] ) )
#define test_index_cycl(ind , Nind, tab, offset ) ( (ind) >= Nind ? tab[ offset (ind - Nind) ] : ( (ind) < 0 ? tab[ offset (Nind + ind) ] : tab[ offset (ind) ] ) )

#define convol_step( ind, Nind, tab, offset ) ( Coeff_h0 * tab[ offset ind ] \
                                              + Coeff_h1 * ( test_index_cycl(ind - step, Nind, tab, offset)     \
                                                           + test_index_cycl(ind + step, Nind, tab, offset) )   \
                                              + Coeff_h2 * ( test_index_cycl(ind - 2*step, Nind, tab, offset)   \
                                                           + test_index_cycl(ind + 2*step, Nind, tab, offset)))

starlet_2d::starlet_2d(int Nx, int Ny, int nscales):
    nscales(nscales), Nx(Nx), Ny(Ny)
{
    // Allocate temporary arrays
    tmpWavelet = new double[Nx * Ny];
    tmpConvol  = new double[Nx * Ny];

    // Set the transform coefficients
    Coeff_h0 = 3. / 8.;
    Coeff_h1 = 1. / 4.;
    Coeff_h2 = 1. / 16.;

    //Computing normalization factor for the wavelet transform
    norm      = new double[nscales];
    norm_gen1 = new double[nscales];
    for (int i = 0; i < nscales; i++) {
        norm_gen1[i] = 1.0;
        norm[i]      = 1.0;
    }

    double *frame = new double[Nx * Ny];
    double *wt    = new double[Nx * Ny * nscales];
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            if (i == Nx / 2 && j == Ny / 2) {
                frame[i * Ny + j] = 1.0;
            } else {
                frame[i * Ny + j] = 0.0;
            }
        }
    }
    transform(frame, wt);
    for (int b = 0; b < nscales; b++) {
        double temp = 0.0;
        for (int i = 0; i < Nx; i++)
            for (int j = 0; j < Ny; j++) {
                temp += wt[b * Nx * Ny + i * Ny + j] * wt[b * Nx * Ny + i * Ny + j];
            }
        norm[b] = sqrt(temp);
    }
    transform_gen1(frame, wt);
    for (int b = 0; b < nscales; b++) {
        double temp = 0.0;
        for (int i = 0; i < Nx; i++)
            for (int j = 0; j < Ny; j++) {
                temp += wt[b * Nx * Ny + i * Ny + j] * wt[b * Nx * Ny + i * Ny + j];
            }
        norm_gen1[b] = sqrt(temp);
    }
    delete[] frame;
    delete[] wt;
}

starlet_2d::~starlet_2d()
{
    delete[] tmpWavelet;
    delete[] tmpConvol;
    delete[] norm;
    delete[] norm_gen1;
}


void starlet_2d::transform(double *In, double *AlphaOut, bool normalized)
{

    // copy delta into first wavelet scale
    for (long ind = 0; ind < Nx * Ny; ind++) {
        AlphaOut[ind] = In[ind];
    }
    for (int s = 0; s < nscales - 1;  s++) {
        long step = (int)(pow((double)2., (double) s) + 0.5);
        ///////////////////////////////////////////////////////////
        // b3spline convolution
        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                tmpConvol[j * Nx + i] = convol_step(j , Ny, AlphaOut, s * Nx * Ny + i + Nx *);
            }

        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                AlphaOut[(s + 1)*Nx * Ny + j * Nx + i] = convol_step(i, Nx, tmpConvol , j * Nx +);
            }
        // End of convolution
        //////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////
        //// SECOND Convolution
        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                tmpConvol[j * Nx + i] = convol_step(j , Ny, AlphaOut, (s + 1) * Nx * Ny + i + Nx *);
            }

        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                AlphaOut[s * Nx * Ny + j * Nx + i] -= convol_step(i, Nx, tmpConvol, j * Nx +);
            }
        ///// End of Convolution
        ////////////////////////////////////////////////////////////
    }

    if (normalized) {
        for (int b = 0; b < nscales; b++) {
            for (long ind = 0; ind < Nx * Ny; ind++) {
                AlphaOut[b * Nx * Ny + ind] /= norm[b];
            }
        }
    }

}

void starlet_2d::transform_gen1(double *In, double *AlphaOut, bool normalized)
{

    // copy delta into first wavelet scale
    for (long ind = 0; ind < Nx * Ny; ind++) {
        AlphaOut[ind] = In[ind];
    }
    for (int s = 0; s < nscales - 1;  s++) {
        long step = (int)(pow((double)2., (double) s) + 0.5);
        ///////////////////////////////////////////////////////////
        // b3spline convolution
        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                tmpConvol[j * Nx + i] = convol_step(j, Ny, AlphaOut, s * Nx * Ny + i + Nx *);
            }

        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                double Val = convol_step(i, Nx, tmpConvol, j * Nx +);
                AlphaOut[(s + 1)*Nx * Ny + j * Nx + i] = Val;
                AlphaOut[s * Nx * Ny + j * Nx + i] -=  Val;
            }
        // End of convolution
        //////////////////////////////////////////////////////////
    }

    if (normalized) {
        for (int b = 0; b < nscales; b++) {
            for (long ind = 0; ind < Nx * Ny; ind++) {
                AlphaOut[b * Nx * Ny + ind] /= norm_gen1[b];
            }
        }
    }
}



void starlet_2d::reconstruct(double *AlphaIn, double *Out, bool normalized)
{
    if (normalized) for (long ind = 0; ind < Nx * Ny; ind++) {
            Out[ind] = AlphaIn[(nscales - 1) * Nx * Ny + ind] * norm[nscales - 1];
        }
    else  for (long ind = 0; ind < Nx * Ny; ind++) {
            Out[ind] = AlphaIn[(nscales - 1) * Nx * Ny + ind];
        }
    for (int s = nscales - 2; s >= 0 ; s--) {
        ///////////////////////////////////////////////////////////
        // b3spline convolution
        int step = (int)(pow((double)2., (double) s) + 0.5);
        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                tmpConvol[j * Nx + i] = convol_step(j, Ny, Out, i + Nx *);
            }

        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                tmpWavelet[j * Nx + i] = convol_step(i, Nx, tmpConvol, j * Nx +);
            }
        // End of convolution
        //////////////////////////////////////////////////////////

        if (normalized) {
            for (int j = 0; j < Ny; j++)
                for (int i = 0; i < Nx; i++) {
                    Out[j * Nx + i] = tmpWavelet[j * Nx + i] +  AlphaIn[s * Nx * Ny + j * Nx + i] * norm[s];
                }
        } else {
            for (int j = 0; j < Ny; j++)
                for (int i = 0; i < Nx; i++) {
                    Out[j * Nx + i] = tmpWavelet[j * Nx + i] +  AlphaIn[s * Nx * Ny + j * Nx + i];
                }
        }
    }
}


void starlet_2d::trans_adjoint(double *AlphaIn, double *Out, bool normalized)
{
    // Add last scale
    //memcpy(pt_delta[z*Nx*Ny],AlphaIn[z][params.nscales2d -1],Nx*Ny*sizeof(double));
    // We are removing the last wavelet scale, so set delta to zero
    if (normalized) for (long ind = 0; ind < Nx * Ny; ind++) {
            Out[ind] = AlphaIn[(nscales - 1) * Nx * Ny + ind] * norm[nscales - 1];    //0.0;
        }
    else  for (long ind = 0; ind < Nx * Ny; ind++) {
            Out[ind] = AlphaIn[(nscales - 1) * Nx * Ny + ind];
        }
    for (int s = nscales - 2; s >= 0 ; s--) {
        ///////////////////////////////////////////////////////////
        // b3spline convolution
        int step = (int)(pow((double)2., (double) s) + 0.5);
        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                tmpConvol[j * Nx + i] = convol_step(j, Ny, Out, i + Nx *);
            }

        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                Out[j * Nx + i] = convol_step(i, Nx, tmpConvol, j * Nx +);
            }
        // End of convolution
        //////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////
        //// SECOND Convolution
        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                tmpConvol[j * Nx + i] = convol_step(j, Ny, AlphaIn, (s) * Nx * Ny + i + Nx *);
            }

        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                tmpWavelet[j * Nx + i] = convol_step(i, Nx, tmpConvol, j * Nx +);
            }
        ///// End of Convolution
        ////////////////////////////////////////////////////////////


        ///////////////////////////////////////////////////////////
        //// THIRD Convolution
        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                tmpConvol[j * Nx + i] = convol_step(j, Ny, tmpWavelet, i + Nx *);
            }

        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                Out[j * Nx + i] +=  AlphaIn[s * Nx * Ny + j * Nx + i] - convol_step(i, Nx, tmpConvol, j * Nx +);
            }
        ///// End of Convolution
        ////////////////////////////////////////////////////////////
    }

}



void starlet_2d::trans_adjoint_gen1(double *AlphaIn, double *Out, bool normalised)
{
    if (normalised) for (long ind = 0; ind < Nx * Ny; ind++) {
            Out[ind] = AlphaIn[(nscales - 1) * Nx * Ny + ind] / norm_gen1[nscales - 1];
        }
    else for (long ind = 0; ind < Nx * Ny; ind++) {
            Out[ind] = AlphaIn[(nscales - 1) * Nx * Ny + ind];
        }
    for (int s = nscales - 2; s >= 0 ; s--) {
        ///////////////////////////////////////////////////////////
        // b3spline convolution
        int step = (int)(pow((double)2., (double) s) + 0.5);
        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                tmpConvol[j * Nx + i] = convol_step(j, Ny, Out, i + Nx *);
            }

        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                Out[j * Nx + i] = convol_step(i, Nx, tmpConvol, j * Nx +);
            }
        // End of convolution
        //////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////
        //// SECOND Convolution
        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                tmpConvol[j * Nx + i] = convol_step(j, Ny, AlphaIn, (s) * Nx * Ny + i + Nx *);
            }

        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                double Val = convol_step(i, Nx, tmpConvol, j * Nx +);
                Out[j * Nx + i] += normalised ? (AlphaIn[(s) * Nx * Ny + j * Nx + i] - (double) Val) / norm_gen1[s] : AlphaIn[(s) * Nx * Ny + j * Nx + i] - (double) Val ;
            }
        ///// End of Convolution
        ////////////////////////////////////////////////////////////
    }
}

void starlet_2d::rec_adjoint(double *In, double *AlphaOut, bool normalized)
{
    // copy delta into first wavelet scale
    for (long ind = 0; ind < Nx * Ny; ind++) {
        AlphaOut[ind] = In[ind];
    }
    for (int s = 0; s < nscales - 1;  s++) {
        long step = (int)(pow((double)2., (double) s) + 0.5);
        ///////////////////////////////////////////////////////////
        // b3spline convolution
        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                tmpConvol[j * Nx + i] = convol_step(j, Ny, AlphaOut, s * Nx * Ny + i + Nx *);
            }

        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++) {
                AlphaOut[(s + 1)*Nx * Ny + j * Nx + i] = convol_step(i, Nx, tmpConvol, j * Nx +);
            }
        // End of convolution
        //////////////////////////////////////////////////////////
    }

    if (normalized) {
        for (int b = 0; b < nscales; b++) {
            for (long ind = 0; ind < Nx * Ny; ind++) {
                AlphaOut[b * Nx * Ny + ind] *= norm[b];
            }
        }
    }
}