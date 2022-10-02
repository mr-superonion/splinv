# Copyright 20220706 Xiangchong Li & Shouzhuo Yang.
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
import h5py
import splinv
import numpy as np
from splinv import hmod
from splinv import detect
from astropy.io import fits
from astropy.table import Table
from splinv import darkmapper
from splinv.grid import Cartesian
from configparser import ConfigParser
import pandas as pd


def extract_valid_values(mass_bias_array):
    '''Mass_bias_array: array based on which you will be extracting valid values'''
    shape = mass_bias_array.shape
    mask = np.full(shape, False)  # where valid values lies
    for i in range(shape[0]):
        for j in range(shape[1]):
            bias_array_masked = np.ma.masked_invalid(mass_bias_array[i, j])
            mask[i, j, :] = ~bias_array_masked.mask
    return mask


def average_bias(value_array, mask):
    shape = value_array.shape
    average_vals = np.zeros((shape[0], shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            average_vals[i, j] = np.average(value_array[i, j][mask[i, j]])
    return average_vals


def plot_biases(filenames, data_name):
    return


def average_across(filenames, data_name):
    """filenames: list of strings for filename
   data_name: string of which parameter you are checking"""
    result = np.array([])
    for filename in filenames:
        file = fits.open(filename)  # default mode is read only
        data = file[1].data
        to_average = data[data_name]
        result = np.append(result, np.average(to_average, axis=2))
        file.close()
    return result


class Simulator:
    """
    A class to perform bias analysis and simulates reconstruction wih splinv.
    1. Simulate either from input data or semi-analytic data generated from halos.
    2. Semi-analytic data can be generate across: halo mass, rs, redshift, ellipticity (denoted as a_over_c).
    3. Semi-analytic data can be noisy or noiseless
    4. It can save reconstructed mass, redshift, and other data from reconstruction
    5. parser... is just a config parser.
    """

    def __init__(self, parser, short_file=False):
        file_name_raw = parser.get('file', 'file_name')
        self.file_name = file_name_raw.split(", ")  # if the file names are separated by ", " this is how to split them
        self.n_a_over_c_sample = parser.getint('simulation', 'n_a_over_c_sample')
        self.init_file_name = parser.get('file', 'init_filename')  # do change the dictionary name
        dictionary_name_raw = parser.get('file', 'dictionary_name')
        self.dictionary_name = dictionary_name_raw.split(", ")
        another_parser = ConfigParser()
        # print(self.init_file_name)
        another_parser.read(self.init_file_name)
        self.Grid = Cartesian(another_parser)
        self.z_samp = self.Grid.zlcgrid  # z of mock data halo
        self.n_z_samp = len(self.z_samp)
        ellipticity_max = parser.getfloat('simulation', 'ellipticity_max')
        ellipticity_min = parser.getfloat('simulation', 'ellipticity_min')
        self.nzl = another_parser.getint('lens', 'nlp')  # how many candidate layers of halos.
        self.nframe = another_parser.getint('sparse', 'nframe')
        if np.abs(ellipticity_max - ellipticity_min) < 0.1:
            self.a_over_c_sample = np.array([ellipticity_max])  # in this case not producing triaxial halos.
            self.n_a_over_c_sample = 1
        else:
            self.a_over_c_sample = np.linspace(ellipticity_min, ellipticity_max, self.n_a_over_c_sample)  # a/c of halos
        if parser.has_option('simulation', 'n_type'):
            self.n_type = parser.getint('simulation', 'n_type')  # types of halo (NFW, alpha=1.5, etc)
        else:
            self.n_type = int(1)
        self.n_trials = parser.getint('simulation', 'n_trials')
        self.short_file = short_file
        # shapes in which data are saved
        self.shape_basic_input = (self.n_z_samp, self.n_a_over_c_sample)  # each trail uses same setup
        self.shape_basic_simulation_result = (self.n_z_samp, self.n_a_over_c_sample, self.n_trials)
        self.shape_detail_input = (
            self.n_z_samp, self.n_a_over_c_sample, self.n_trials, self.nzl, self.Grid.ny, self.Grid.nx)
        self.shape_detail_simulation_result = (
            self.n_z_samp, self.n_a_over_c_sample, self.n_trials, self.nzl, self.nframe, self.Grid.ny,
            self.Grid.nx)
        self.noise_std = self.prepare_noise_std()
        if parser.has_option('false_detection', 'enable_false_detection'):
            self.enable_false_detection = parser.getboolean('false_detection', 'enable_false_detection')
            self.distance_limit = parser.getint('false_detection', 'distance_limit')
            self.redshift_limit = parser.getint('false_detection', 'redshift_limit')
        else:
            self.enable_false_detection = False

    def create_files_h5(self):
        """Not that supported under parallelization"""
        n_z_samp = self.n_z_samp
        n_trials = self.n_trials
        n_a_over_c_sample = self.n_a_over_c_sample
        n_type = self.n_type
        n_frame = self.nframe

        # shapes needed for file setup
        shape_basic_input = (n_z_samp, n_a_over_c_sample)  # each trail uses same setup
        shape_basic_simulation_result = (n_z_samp, n_a_over_c_sample, n_trials)
        shape_detail_input = (n_z_samp, n_a_over_c_sample, n_trials, self.nzl, self.Grid.ny, self.Grid.nx)
        shape_detail_simulation_result = (n_z_samp, n_a_over_c_sample, n_trials, self.nzl, n_frame, self.Grid.ny,
                                          self.Grid.nx)
        if self.short_file:
            for name in self.file_name:
                file = h5py.File(name, 'w')
                file_basics_group = file.create_group('basics')
                # each trial has same input redshift and a_over_c
                input_redshift = file_basics_group.create_dataset(name='input_redshift',
                                                                  shape=shape_basic_input,
                                                                  dtype=np.float32)
                input_a_over_c = file_basics_group.create_dataset(name='input_a_over_c',
                                                                  shape=shape_basic_input,
                                                                  dtype=np.float32)
                true_mass = file_basics_group.create_dataset(name='true_mass', shape=1,
                                                             dtype=np.float64)  # only 1 true mass
                simulated_mass = file_basics_group.create_dataset(name='simulated_mass',
                                                                  shape=shape_basic_simulation_result,
                                                                  dtype=np.float64)
                same_redshift = file_basics_group.create_dataset(name='same_redshift',
                                                                 shape=shape_basic_simulation_result,
                                                                 dtype=bool)
                mass_bias = file_basics_group.create_dataset(name='mass_bias', shape=shape_basic_simulation_result,
                                                             dtype=np.float64)
                simulated_redshift = file_basics_group.create_dataset(name='simulated_redshift',
                                                                      shape=shape_basic_simulation_result,
                                                                      dtype=np.float32)
        else:
            for name in self.file_name:
                file = h5py.File(name, 'w')
                file_basics_group = file.create_group('basics')
                # each trial has same input redshift and a_over_c
                input_redshift = file_basics_group.create_dataset(name='input_redshift',
                                                                  shape=shape_basic_input,
                                                                  dtype=np.float32)
                input_a_over_c = file_basics_group.create_dataset(name='input_a_over_c',
                                                                  shape=shape_basic_input,
                                                                  dtype=np.float32)
                true_mass = file_basics_group.create_dataset(name='true_mass', shape=1,
                                                             dtype=np.float64)  # only 1 true mass
                simulated_mass = file_basics_group.create_dataset(name='simulated_mass',
                                                                  shape=shape_basic_simulation_result,
                                                                  dtype=np.float64)
                same_redshift = file_basics_group.create_dataset(name='same_redshift',
                                                                 shape=shape_basic_simulation_result,
                                                                 dtype=bool)
                mass_bias = file_basics_group.create_dataset(name='mass_bias', shape=shape_basic_simulation_result,
                                                             dtype=np.float64)
                simulated_redshift = file_basics_group.create_dataset(name='simulated_redshift',
                                                                      shape=shape_basic_simulation_result,
                                                                      dtype=np.float32)
                file_detail_group = file.create_group('detail')
                # shape might be ny,nx but it doesn't matter for now....
                dmapper_w = file_detail_group.create_dataset(name='dmapper_w',
                                                             shape=shape_detail_simulation_result,
                                                             dtype=np.float32)
                alpha_R = file_detail_group.create_dataset(name='alphaR',
                                                           shape=shape_detail_simulation_result,
                                                           dtype=np.float32)
                input_shear = file_detail_group.create_dataset(name='input_shear',
                                                               shape=shape_detail_input,
                                                               dtype=np.complex64)
                file.close()

    def create_files_fits(self):
        # pretty self-explanatory, I think
        input_redshift = np.zeros(self.shape_basic_input)
        input_a_over_c = np.zeros(self.shape_basic_input)
        true_mass = np.zeros(self.shape_basic_input)
        simulated_mass = np.zeros(self.shape_basic_simulation_result)
        same_redshift = np.zeros(self.shape_basic_simulation_result, dtype=bool)
        mass_bias = np.zeros(self.shape_basic_simulation_result)
        simulated_redshift = np.zeros(self.shape_basic_simulation_result)
        dmapper_w = np.zeros(self.shape_detail_simulation_result)
        alpha_R = np.zeros(self.shape_detail_simulation_result)
        input_shear = np.zeros(self.shape_detail_input, dtype=np.complex64)
        dim_basic_input = str(self.shape_basic_input[1:][::-1])
        n_basic_input = str(np.prod(self.shape_basic_input[1:]))
        dim_basic_simulation_result = str(self.shape_basic_simulation_result[1:][::-1])
        n_basic_simulation_result = str(np.prod(self.shape_basic_simulation_result[1:]))
        dim_detail_input = str(self.shape_detail_input[1:][::-1])
        n_detail_input = str(np.prod(self.shape_detail_input[1:]))
        dim_detail_simulation_result = str(self.shape_detail_simulation_result[1:][::-1])
        n_detail_simulation_result = str(np.prod(self.shape_detail_simulation_result[1:]))
        if self.short_file:
            for name in self.file_name:
                c1 = fits.Column(name='input_redshift', array=input_redshift, format=n_basic_input + 'E',
                                 dim=dim_basic_input)
                c2 = fits.Column(name='input_a_over_c', array=input_a_over_c, format=n_basic_input + 'E',
                                 dim=dim_basic_input)
                c3 = fits.Column(name='true_mass', array=true_mass, format=n_basic_input + 'D', dim=dim_basic_input)
                c4 = fits.Column(name='simulated_mass', array=simulated_mass, format=n_basic_simulation_result + 'D',
                                 dim=dim_basic_simulation_result)
                c5 = fits.Column(name='same_redshift', array=same_redshift, format=n_basic_simulation_result + 'L',
                                 dim=dim_basic_simulation_result)
                c6 = fits.Column(name='mass_bias', array=mass_bias, format=n_basic_simulation_result + 'D',
                                 dim=dim_basic_simulation_result)
                c7 = fits.Column(name='simulated_redshift', array=simulated_redshift,
                                 format=n_basic_simulation_result + 'E', dim=dim_basic_simulation_result)
                t = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7])
                t.writeto(name, overwrite=True)
        else:
            for name in self.file_name:
                c1 = fits.Column(name='input_redshift', array=input_redshift, format=n_basic_input + 'E',
                                 dim=dim_basic_input)
                c2 = fits.Column(name='input_a_over_c', array=input_a_over_c, format=n_basic_input + 'E',
                                 dim=dim_basic_input)
                c3 = fits.Column(name='true_mass', array=true_mass, format=n_basic_input + 'D', dim=dim_basic_input)
                c4 = fits.Column(name='simulated_mass', array=simulated_mass, format=n_basic_simulation_result + 'D',
                                 dim=dim_basic_simulation_result)
                c5 = fits.Column(name='same_redshift', array=same_redshift, format=n_basic_simulation_result + 'L',
                                 dim=dim_basic_simulation_result)
                c6 = fits.Column(name='mass_bias', array=mass_bias, format=n_basic_simulation_result + 'D',
                                 dim=dim_basic_simulation_result)
                c7 = fits.Column(name='simulated_redshift', array=simulated_redshift,
                                 format=n_basic_simulation_result + 'E', dim=dim_basic_simulation_result)
                c8 = fits.Column(name='dmapper_w', array=dmapper_w, format=n_detail_simulation_result + 'D',
                                 dim=dim_detail_simulation_result)
                c9 = fits.Column(name='alpha_R', array=alpha_R, format=n_detail_simulation_result + 'D',
                                 dim=dim_detail_simulation_result)
                c10 = fits.Column(name='input_shear', array=input_shear, format=n_detail_input + 'C',
                                  dim=dim_detail_input)
                t = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
                t.writeto(name, overwrite=True)

    def prepare_argument(self, halo_masses, halo_types, lbd, noise):
        """
        Caution: Right now it does not support multiple types of halo yet.
        :param halo_masses: an array of log masses
        :param halo_types: a list of strings dictating halo types
        :param lbd: an array of lbd in sparse reconstruction
        :param noise: whether noisy construction
        :return: the arguments to start multipool processing
        """
        arguments = []
        if not len(halo_masses) == len(self.file_name):
            raise ValueError('there should be as many files as there are masses')
        if not len(halo_masses) == len(lbd):
            raise ValueError('there should be as many mass as there are lbds')
        if not len(halo_masses) == len(halo_types):
            raise ValueError('halo types correspond to halo shapes')
        # all simulation
        for i in range(len(halo_masses)):
            # iterating through files, which uses same lbd, mass, and types of halo
            for j in range(self.n_z_samp):
                for k in range(self.n_a_over_c_sample):
                    for l in range(self.n_trials):
                        # which number of trials we are on
                        arguments.append([self.dictionary_name[i], halo_masses[i], lbd[i], self.file_name[i],
                                          halo_types[i], j, k, l, noise[i]])
        return arguments

    def prepare_noise_std(self):
        halo = hmod.triaxialJS02(mass=1, conc=4, redshift=0.2425, ra=0., dec=0., a_over_b=1,
                                 a_over_c=1, tri_nfw=True,
                                 long_truncation=True, OLS03=True)  # just a placeholder halo
        another_parser = ConfigParser()  # parser for reconstruction
        another_parser.read(self.init_file_name)  # make sure noise is created with same smoothing etc.
        general_grid = splinv.hmod.triaxialJS02_grid_mock(another_parser)
        all_noise = np.zeros((100, 10, 48, 48), dtype=np.complex64)
        for i in range(100):
            all_noise[i, :, :, :] = general_grid.calc_noise(halo)
        dg1 = all_noise.real
        dg2 = all_noise.imag
        noise_std = np.zeros((10, 48, 48))
        for l in range(48):
            for m in range(48):
                for n in range(10):
                    noise_std[n, l, m] = np.sqrt(np.std(dg1[:, n, l, m]) ** 2 + np.std(dg2[:, n, l, m]) ** 2)
        return noise_std

    def simulate(self, args):
        """
        :param args contains the following (and it is a list).
        :param dictionary_name: which file to use as dictionary
        :param log_m: log mass
        :param lbd: lbd in lasso
        :param save_file_name: open this file and write in it
        :param halo_type: nfw or cuspy
        :param z_index:
        :param a_over_c_index:
        :param trial_index: which number of realization on (later to take average).
        :return: write in files.
        """
        # Parsing argument
        dictionary_name = args[0]
        log_m = args[1]
        lbd = args[2]
        save_file_name = args[3]
        halo_type = args[4]
        z_index = args[5]
        a_over_c_index = args[6]
        trial_index = args[7]
        noise = args[8]

        z_h = self.z_samp[z_index]
        a_over_c = self.a_over_c_sample[a_over_c_index]
        tri_nfw = False
        if halo_type == 'nfw':
            tri_nfw = True
            print('nfw')
        else:
            print('cuspy')
        M_200 = 10. ** log_m
        conc = 4
        halo = hmod.triaxialJS02(mass=M_200, conc=conc, redshift=z_h, ra=0., dec=0., a_over_b=1,
                                 a_over_c=a_over_c, tri_nfw=tri_nfw,
                                 long_truncation=True, OLS03=True)
        another_parser = ConfigParser()  # parser for reconstruction
        another_parser.read(self.init_file_name)
        another_parser.set('lens', 'SigmaFname', dictionary_name)
        # file = h5py.File(save_file_name, 'r+')
        # file['basics/input_redshift'][z_index,a_over_c_index] = z_h
        # file['basics/input_a_over_c'][z_index, a_over_c_index] = a_over_c
        # now... only has capacity of 1 scale radius
        Grid = Cartesian(another_parser)
        lensKer1 = Grid.lensing_kernel(deltaIn=False)
        general_grid = splinv.hmod.triaxialJS02_grid_mock(another_parser)
        if noise:
            data2, gErrval = general_grid.add_halo_from_dsigma(halo, add_noise=True,
                                                               seed=trial_index)  # add same random seed
            gErr = self.noise_std
            print('noisy reconstruction')
        else:
            data2 = general_grid.add_halo(halo)[1]
            gErrval = 0.05
            gErr = np.ones(Grid.shape) * gErrval
            print('noiseless reconstruction')
        # gErr = np.ones(Grid.shape) * gErrval
        # file['detail/input_shear'][z_index, a_over_c_index, trial_index, trial_index, :, :, :] = data2
        # file['basics/true_mass'] = M_200
        dmapper = darkmapper(another_parser, data2.real, data2.imag, gErr, lensKer1)
        dmapper.lbd = lbd  # Lasso penalty.
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
        c1 = detect.local_maxima_3D(dmapper.deltaR)[0]  # the peak value is not important
        print(c1)
        ndet = c1.shape[0]
        print('Detected %d clusters!' % ndet)
        z_col = c1[:, 0]
        y_col = c1[:, 1]
        x_col = c1[:, 2]
        x_center = np.ones_like(y_col) * 24.
        y_center = np.ones_like(y_col) * 24.

        mass_est = np.zeros_like(z_col, dtype=np.float128)
        frame_counter = np.zeros_like(z_col, dtype=int)
        for i in range(ndet):
            for j in range(self.nframe):
                mass_at_loc = (dmapper.alphaR * dmapper._w)[z_col[i], j, y_col[i], x_col[i]]
                if mass_at_loc > 10:  # mass detected
                    frame_counter = frame_counter + 2 ** j
                mass_est[i] = mass_est[i] + mass_at_loc
        log_m_est = np.log10(mass_est) + 14.
        # original halo info
        input_z_index = np.ones_like(z_col, dtype=int) * z_index
        input_a_over_c_index = np.ones_like(z_col, dtype=int) * a_over_c_index
        halo_id = np.ones_like(z_col, dtype=int) * int(trial_index)

        distance_from_center = np.sqrt((x_col - x_center) ** 2 + (y_col - y_center) ** 2)
        redshift_bias = np.abs(z_col - input_z_index)
        print('distance', distance_from_center)
        print('bias', redshift_bias)
        valid_distance = np.ma.masked_less_equal(distance_from_center, 3).mask
        valid_redshift = np.ma.masked_less_equal(redshift_bias, 2).mask
        print('valid distance', valid_distance)
        print('valid_redshift', valid_redshift)
        successful_reconstruction = np.logical_and(valid_distance, valid_redshift)  # important info on successful recon

        # out_table = Table()
        # # XL:
        # # for each simulation, we write a detected catalog to disk
        # # this is also what we do in real observation -- read shear map and
        # # generate halo catalog and write to disk
        # out_table['z'] = z_col
        # out_table['y'] = y_col
        # out_table['x'] = x_col
        # out_table['log10m'] = log_m_est

        os.makedirs(save_file_name, exist_ok=True)
        df = pd.DataFrame({'reconstructed_z': z_col,
                           'reconstructed_x': x_col,
                           'reconstructed_y': y_col,
                           'reconstructed_log10m': log_m_est,
                           'input_z': input_z_index,
                           'input_a_over_c': input_a_over_c_index,
                           'successful_reconstruction': successful_reconstruction.astype(int),
                           'halo_id': halo_id})
        file_name = str(z_index) + str(a_over_c_index) + str(trial_index) + '.csv'
        df.to_csv(save_file_name + '/' + file_name, index=False)
        # save_file_name='outputs/src_z%02d_m%02d_n%03d.fits' %(sim_zid,sim_mid,sim_nid)
        # out_table.write()

        # XL:
        # the following parameters can be derived from the detected cluster
        # catalog and the saved parameters of input halo, we do not need to
        # save them:
        # z_index, a_over_c_index, trial_index
        # M_200, same_redshift, simulated_redshift, simulated_mass,
        # mass_bias, dmapper.alphaR, dmapper._w

        # XL: data2 is not useful!
        # data2

        # try:
        #     simulated_redshift_index = c1[0][0] # need to save all the elements
        #     simulated_redshift = self.z_samp[simulated_redshift_index]
        #     if simulated_redshift_index != z_index:
        #         # file['basics/same_redshift'][z_index, a_over_c_index, trial_index] = False
        #         same_redshift = False
        #     else:
        #         # file['basics/same_redshift'][z_index, a_over_c_index, trial_index] = True
        #         same_redshift = True
        #     print('stop here!!')
        #     return

        #     # XL: I do not think the following code does the job you want so comment it out
        #     # frame_index = 0
        #     # for i in range(self.nframe):
        #     #     location = detect.local_maxima_3D(dmapper.alphaR[:, i, :, :])
        #     #     if len(location[0]) > 0:  # max detected
        #     #         frame_index = i  # which "frame"
        #     #         c1 = location  # redshift plane, and position
        #     #         break
        #     # See the example below
        #     reconstructed_log_m = np.log10(
        #         (dmapper.alphaR * dmapper._w)[c1[0][0][0], frame_index, c1[0][0][1], c1[0][0][2]]) + 14.
        # except:
        #     print('detection failed')
        #     reconstructed_log_m = -np.inf
        #     same_redshift = False
        #     simulated_redshift = -np.inf  # impossible value

        # file['basics/simulated_mass'][
        #     z_index, a_over_c_index, trial_index] = 10 ** reconstructed_log_m
        # simulated_mass = 10 ** reconstructed_log_m
        # if reconstructed_log_m > 12:  # otherwise considered as a failed reconstruction
        #     # file['basics/mass_bias'][
        #     #     z_index, a_over_c_index, trial_index] = M_200 - 10 ** reconstructed_log_m
        #     mass_bias = M_200 - 10 ** reconstructed_log_m
        # else:
        #     # file['basics/mass_bias'][z_index, a_over_c_index, trial_index] = -np.inf
        #     mass_bias = -np.inf
        #     # just giving it a non-readable value for plotting
        # file['detail/alpha_R'][z_index, a_over_c_index, trial_index, :, :, :, :] = dmapper.alphaR
        # file['detail/dmapper_w'][z_index, a_over_c_index, trial_index, :, :, :, :] = dmapper._w
        # file.close()
        return

    def write_files_fits(self, outputs):
        """:param outputs: list of outputs"""
        if self.short_file:
            for output in outputs:
                # this is going to be a very long for loop, but it cannot be help
                file = fits.open(output[-1], mode='update')
                data = file[1].data
                z_index = output[0]
                a_over_c_index = output[1]
                trial_index = output[2]
                data['input_redshift'][z_index, a_over_c_index] = self.z_samp[output[0]]
                data['input_a_over_c'][z_index, a_over_c_index] = self.a_over_c_sample[output[1]]
                data['true_mass'][z_index, a_over_c_index] = output[4]
                data['simulated_mass'][z_index, a_over_c_index, trial_index] = output[7]
                data['same_redshift'][z_index, a_over_c_index, trial_index] = output[5]
                data['mass_bias'][z_index, a_over_c_index, trial_index] = output[8]
                data['simulated_redshift'][z_index, a_over_c_index, trial_index] = output[6]
                print('writing data in')
                file.close()
        else:
            for output in outputs:
                # this is going to be a very long for loop, but it cannot be help
                file = fits.open(output[-1], mode='update')
                data = file[1].data
                z_index = output[0]
                a_over_c_index = output[1]
                trial_index = output[2]
                data['input_redshift'][z_index, a_over_c_index] = self.z_samp[output[0]]
                data['input_a_over_c'][z_index, a_over_c_index] = self.a_over_c_sample[output[1]]
                data['true_mass'][z_index, a_over_c_index] = output[4]
                data['simulated_mass'][z_index, a_over_c_index, trial_index] = output[7]
                data['same_redshift'][z_index, a_over_c_index, trial_index] = output[5]
                data['mass_bias'][z_index, a_over_c_index, trial_index] = output[8]
                data['simulated_redshift'][z_index, a_over_c_index, trial_index] = output[6]
                data['dmapper_w'][z_index, a_over_c_index, trial_index, :, :, :, :] = output[10]  # not necessary
                data['alpha_R'][z_index, a_over_c_index, trial_index, :, :, :, :] = output[9]
                data['input_shear'][z_index, a_over_c_index, trial_index, :, :, :] = output[3]
                print('writing data in')
                file.close()
