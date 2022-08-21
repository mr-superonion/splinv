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


class Simulator:
    """
    A class to perform bias analysis and simulates reconstruction wih splinv.
    1. Simulate either from input data or semi-analytic data generated from halos.
    2. Semi-analytic data can be generate across: halo mass, rs, redshift, ellipticity (denoted as a_over_c).
    3. Semi-analytic data can be noisy or noiseless
    4. It can save reconstructed mass, redshift, and other data from reconstruction
    5. parser... is just a config parser.
    """

    def __init__(self, parser):
        file_name_raw = parser.get('file', 'file_name')
        self.file_name = file_name_raw.split(", ")  # if the file names are separated by ", " this is how to split them
        self.n_a_over_c_sample = parser.getint('simulation', 'n_a_over_c_sample')
        self.init_file_name = parser.get('file', 'init_filename')  # do change the dictionary name
        dictionary_name_raw = parser.get('file', 'dictionary_name')
        self.dictionary_name = dictionary_name_raw.split(", ")
        another_parser = ConfigParser()
        #print(self.init_file_name)
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

    def create_files(self):
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
            same_redshift = file_basics_group.create_dataset(name='same_redshift', shape=shape_basic_simulation_result,
                                                             dtype=bool)
            mass_bias = file_basics_group.create_dataset(name='mass_bias', shape=shape_basic_simulation_result,
                                                         dtype=np.float64)
            simulated_redshift = file_basics_group.create_dataset(name='simulated_redshift',
                                                                  shape=shape_basic_simulation_result, dtype=np.float32)
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
        file = h5py.File(save_file_name, 'r+')
        file['basics/input_redshift'][z_index,a_over_c_index] = z_h
        file['basics/input_a_over_c'][z_index, a_over_c_index] = a_over_c
        # now... only has capacity of 1 scale radius
        Grid = Cartesian(another_parser)
        lensKer1 = Grid.lensing_kernel(deltaIn=False)
        general_grid = splinv.hmod.triaxialJS02_grid_mock(another_parser)
        if noise:
            data2, gErrval = general_grid.add_halo_from_dsigma(halo, add_noise=True)
            print('noisy reconstruction')
        else:
            data2 = general_grid.add_halo(halo)[1]
            gErrval = 0.05
            print('noiseless reconstruction')
        gErr = np.ones(Grid.shape) * gErrval
        file['detail/input_shear'][z_index, a_over_c_index, trial_index, trial_index, :, :, :] = data2
        file['basics/true_mass'] = M_200
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
        c1 = detect.local_maxima_3D(dmapper.deltaR)

        try:
            if c1[0][0][0] != z_index:
                file['basics/same_redshift'][z_index, a_over_c_index, trial_index] = False
            else:
                file['basics/same_redshift'][z_index, a_over_c_index, trial_index] = True
            file['basics/simulated_redshift'][z_index, a_over_c_index, trial_index] = \
            self.z_samp[c1[0][0][0]]
            reconstructed_log_m = np.log10(
                (dmapper.alphaR * dmapper._w)[c1[0][0][0], 0, c1[0][0][1], c1[0][0][2]]) + 14.
        except:
            print('error detected')
            reconstructed_log_m = -np.inf
        file['basics/simulated_mass'][
            z_index, a_over_c_index, trial_index] = 10 ** reconstructed_log_m
        if reconstructed_log_m > 12:  # otherwise considered as a failed reconstruction
            file['basics/mass_bias'][
                z_index, a_over_c_index, trial_index] = M_200 - 10 ** reconstructed_log_m
        else:
            file['basics/mass_bias'][z_index, a_over_c_index, trial_index] = -np.inf
            # just giving it a non-readable value for plotting
        file['detail/alpha_R'][z_index, a_over_c_index, trial_index, :, :, :, :] = dmapper.alphaR
        file['detail/dmapper_w'][z_index, a_over_c_index, trial_index, :, :, :, :] = dmapper._w
        del dmapper
        file.close()
        return
