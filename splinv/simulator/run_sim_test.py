import splinv
import numpy as np
import h5py
from configparser import ConfigParser
from splinv import utility
from schwimmbad import MPIPool

configName = 'prepare_simulator.ini'
parser = ConfigParser()
parser.read(configName)
simulator = utility.Simulator(parser)
simulator.create_files_fits()
args = simulator.prepare_argument(np.array([14.8,14.8]), ['nfw','cuspy'], np.array([4,4]), [True, False])

pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)
results = pool.map(simulator.simulate, args)
output =   np.stack(results)
simulator.write_files_fits(output)


