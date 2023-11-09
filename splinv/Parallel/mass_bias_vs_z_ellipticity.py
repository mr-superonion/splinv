import numpy as np
from splinv import detect
from splinv import hmod
from splinv import darkmapper
from splinv.grid import Cartesian
from configparser import ConfigParser
import splinv
import h5py
# import time
from schwimmbad import MPIPool
import sys


def do_process(refs):
    parser = ConfigParser()
    parser.read(refs[0])
    parser.set('sparse', 'mu', '3e-4')  # step size for gradient descent
    parser.set('lens', 'resolve_lim', '0.02')  # pix
    parser.set('sparse', 'nframe', '1')
    Grid = Cartesian(parser)
    lensKer1 = Grid.lensing_kernel(deltaIn=False)
    z_samp = Grid.zlcgrid
    n_z_samp = len(z_samp)
    a_over_c_sample = np.linspace(1, 0.4, 10)
    n_a_over_c_sample = len(a_over_c_sample)
    file = h5py.File(refs[3], 'w')
    file_basics_group = file.create_group('basics')
    input_redshift = file_basics_group.create_dataset(name='input_redshift', shape=(n_z_samp, n_a_over_c_sample),
                                                      dtype=np.float64)
    input_a_over_c = file_basics_group.create_dataset(name='input_a_over_c', shape=(n_z_samp, n_a_over_c_sample),
                                                      dtype=np.float64)
    true_mass = file_basics_group.create_dataset(name='true_mass', shape=(n_z_samp, n_a_over_c_sample),
                                                 dtype=np.float64)
    simulated_mass = file_basics_group.create_dataset(name='simulated_mass', shape=(n_z_samp, n_a_over_c_sample),
                                                      dtype=np.float64)
    same_redshift = file_basics_group.create_dataset(name='same_redshift', shape=(n_z_samp, n_a_over_c_sample),
                                                     dtype=bool)
    mass_bias = file_basics_group.create_dataset(name='mass_bias', shape=(n_z_samp, n_a_over_c_sample),
                                                 dtype=np.float64)
    simulated_redshift = file_basics_group.create_dataset(name='simulated_redshift',
                                                          shape=(n_z_samp, n_a_over_c_sample), dtype=np.float64)
    file_detail_group = file.create_group('detail')
    input_sigma = file_detail_group.create_dataset(name='input_sigma',
                                                   shape=(n_z_samp, n_a_over_c_sample, Grid.nx, Grid.ny),
                                                   dtype=np.float64)

    # shape might be ny,nx but it doesn't matter for now....
    dmapper_w = file_detail_group.create_dataset(name='dmapper_w',
                                                 shape=(n_z_samp, n_a_over_c_sample, 10, 1, Grid.nx, Grid.ny),
                                                 dtype=np.float64)
    delta_R = file_detail_group.create_dataset(name='deltaR', shape=(n_z_samp, n_a_over_c_sample, 10, Grid.nx, Grid.ny),
                                               dtype=np.float64)
    alpha_R = file_detail_group.create_dataset(name='alphaR',
                                               shape=(n_z_samp, n_a_over_c_sample, 10, 1, Grid.nx, Grid.ny),
                                               dtype=np.float64)
    aframes = file_detail_group.create_dataset(name='aframes',
                                               shape=(n_z_samp, n_a_over_c_sample, 10, 1, Grid.nx, Grid.ny),
                                               dtype=np.complex128)

    def detect_mass(z_index, a_over_c_index, lbd, log_m, tri_nfw):
        z_h = z_samp[z_index]
        input_redshift[z_index, a_over_c_index] = z_h
        M_200 = 10. ** (log_m)
        true_mass[z_index, a_over_c_index] = M_200
        input_a_over_c[z_index, a_over_c_index] = a_over_c_sample[a_over_c_index]

        conc = 4  # for consistency with NFWshearlet  #6.02*(M_200/1.E13)**(-0.12)*(1.47/(1.+z_h))**(0.16) #https://wwwmpa.mpa-garching.mpg.de/HydroSims/Magneticum/Preprints/cluster_mc.pdf
        halo = hmod.triaxialJS02(mass=M_200, conc=conc, redshift=z_h, ra=0., dec=0., a_over_b=1,
                                 a_over_c=a_over_c_sample[a_over_c_index], tri_nfw=tri_nfw,
                                 long_truncation=True, OLS03=True)
        parser.set('lens', 'rs_base', '%s' % halo.rs)
        Grid = Cartesian(parser)
        lensKer1 = Grid.lensing_kernel(deltaIn=False)
        general_grid = splinv.hmod.triaxialJS02_grid_mock(parser)
        data1 = general_grid.add_halo(halo)[2]  # sigma

        input_sigma[z_index, a_over_c_index, :, :] = data1

        data2 = general_grid.add_halo(halo)[1]
        gErr = np.ones(Grid.shape) * 0.05
        dmapper = darkmapper(parser, data2.real, data2.imag, gErr, lensKer1)
        dmapper.lbd = lbd  # Lasso penalty
        dmapper.lcd = 0.  # Ridge penalty in Elastic net
        dmapper.nonNeg = True  # using non-negative Lasso
        dmapper.clean_outcomes()
        dmapper.fista_gradient_descent(3000)  # run 3000 steps
        w = dmapper.adaptive_lasso_weight(gamma=2.)  # determine the apaptive weight
        dmapper.fista_gradient_descent(3000, w=w)  # run adaptive lasso
        dmapper.mu = 3e-3  # step size for gradient descent
        for _ in range(3):  # redo apaptive lasso
            w = dmapper.adaptive_lasso_weight(gamma=2.)
            dmapper.fista_gradient_descent(3000, w=w)
        dmapper.reconstruct()
        c1 = detect.local_maxima_3D(dmapper.deltaR)
        # if c1[0][0][0] != z_index:
        #     same_redshift[z_index,a_over_c_index] = False
        # else:
        #     same_redshift[z_index,a_over_c_index] = True
        # simulated_redshift[z_index,a_over_c_index]  = z_samp[c1[0][0][0]]
        # reconstructed_log_m = np.log10((dmapper.alphaR*dmapper._w)[c1[0][0][0],0,c1[0][0][1],c1[0][0][2]])+14.

        try:
            if c1[0][0][0] != z_index:
                same_redshift[z_index, a_over_c_index] = False
            else:
                same_redshift[z_index, a_over_c_index] = True
            simulated_redshift[z_index, a_over_c_index] = z_samp[c1[0][0][0]]
            reconstructed_log_m = np.log10(
                (dmapper.alphaR * dmapper._w)[c1[0][0][0], 0, c1[0][0][1], c1[0][0][2]]) + 14.
        except:
            print('error detected')
            reconstructed_log_m = -np.inf
        simulated_mass[z_index, a_over_c_index] = 10 ** reconstructed_log_m
        if 10 ** reconstructed_log_m > 10 ** 13:
            mass_bias[z_index, a_over_c_index] = M_200 - 10 ** reconstructed_log_m
        else:
            mass_bias[z_index, a_over_c_index] = -np.inf  ###just giving it a non-readable value for plotting
        delta_R[z_index, a_over_c_index, :, :, :] = dmapper.deltaR
        alpha_R[z_index, a_over_c_index, :, :, :,
        :] = dmapper.alphaR  # alphafile_detail_group.create_dataset(name = 'alphaR',shape=(n_z_samp,n_a_over_c_sample,10,1,Grid.nx,Grid.ny),dtype=np.float64)
        aframes[z_index, a_over_c_index, :, :, :, :] = dmapper.modelDict.aframes
        dmapper_w[z_index, a_over_c_index, :, :, :, :] = dmapper._w
        del dmapper
        return  # the last one is redshift estimates

    for i in range(2):
        for j in range(2):
            print('i is:', i)
            print('j is:', j)
            lbd = int(refs[2])
            log_m = float(refs[1])
            tri_nfw = bool(refs[4])
            if tri_nfw:
                print('this one is nfw')
            detect_mass(i, j, lbd, log_m, tri_nfw)
    return


def main():
    # start = time.perf_counter()
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     # refs = [['JS02146.ini', 14.6, 4, 'Aug12_146_4_cuspy.h5','False'],
    #     #         ['JS02148.ini', 14.8, 8, 'Aug12_148_8_cuspy.h5','False'],
    #     #         ['JS02148.ini', 14.8, 4, 'Aug12_148_4_cuspy.h5','False'],
    #     #         ['WB00146.ini', 14.6, 4, 'Aug12_146_4_nfw.h5', 'True'],
    #     #         ['WB00148.ini', 14.8, 8, 'Aug12_148_8_nfw.h5', 'True'],
    #     #         ['WB00148.ini', 14.8, 4, 'Aug12_148_4_nfw.h5', 'True']
    #     #         ]
    #     refs = [['JS02146.ini', 14.6, 4, 'Aug12_146_4_cuspy.h5', 'False'],
    #             ['JS02148.ini', 14.8, 8, 'Aug12_148_8_cuspy.h5', 'False'],
    #             ['JS02148.ini', 14.8, 4, 'Aug12_148_4_cuspy.h5', 'False']
    #             ]
    #     results = executor.map(do_process, refs)

    #     # for result in results:
    #     print(result)
    #
    # finish = time.perf_counter()

    # print(f'Finished in {round(finish - start, 2)} second(s)')
    # time_start = time.time()
    refs = [['JS02146.ini', 14.6, 4, 'Aug18_146_4_cuspy.h5', 'False'],
            ['JS02148.ini', 14.8, 8, 'Aug18_148_8_cuspy.h5', 'False'],
            ['JS02148.ini', 14.8, 4, 'Aug18_148_4_cuspy.h5', 'False'],
            ['WB00146.ini', 14.6, 4, 'Aug18_146_4_nfw.h5', 'True'],
            ['WB00148.ini', 14.8, 8, 'Aug18_148_8_nfw.h5', 'True'],
            ['WB00148.ini', 14.8, 4, 'Aug18_148_4_nfw.h5', 'True']
            ]
    # do_process(refs)
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    results = pool.map(do_process, refs)
    # print(time.time() - time_start)
    return


if __name__ == '__main__':
    main()