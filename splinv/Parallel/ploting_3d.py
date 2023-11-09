import numpy as np
from splinv import detect
from splinv import hmod
from splinv import darkmapper
from splinv.grid import Cartesian
from configparser import ConfigParser
import splinv
from astropy.io import fits

from splinv.hmod import triaxialJS02_grid_mock

configName = 'test_mock_catalog_atom-smaller field.ini'
parser = ConfigParser()
parser.read(configName)

# halo simulation
z_h = 0.2425
log_m = 14.745
M_200 = 10. ** (log_m)
conc = 4.
halo = hmod.nfwTJ03(mass=M_200, conc=conc, redshift=z_h, ra=0., dec=0.)
parser.set('sparse', 'mu', '3e-4')  # step size for gradient descent
parser.set('lens', 'resolve_lim', '0.02')  # pix
parser.set('lens', 'rs_base', '%s' % halo.rs)  # Mpc/h
parser.set('sparse', 'nframe', '1')

# Pixelation
Grid = Cartesian(parser)
lensKer1 = Grid.lensing_kernel(deltaIn=False)
# general_grid    =   triaxialJS02_grid_mock(parser)
# data2   =   general_grid.add_halo(halo)[1]
# data1   = general_grid.add_halo(halo)[0]
data2 = fits.getdata('data2.fits')
data3 = fits.getdata('data3.fits')
data1 = fits.getdata('data1.fits')
gErr = np.ones(Grid.shape) * 0.05

z_list = np.linspace(0, 2.5, 30)


def Gaussian_filter(x, mu, sigma, amplitude=1.0):
    return amplitude * 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / (sigma)) ** 2)


import plotly

print(data1.shape)
import plotly.graph_objects as go

data4 = np.ones_like(data3)
z, ra, dec = np.mgrid[0.05:2.5:19j, -10:10:192j,-10:10:192j] #np.mgrid[0.05:2.5:19j, -10:10:96j, -10:10:96j]  # np.mgrid[0.05:2.5:19j, -10:10:192j,-10:10:192j]
for i in range(19):
    data2[i] = data2[i] * (Gaussian_filter(Grid.zcgrid[i], 0.75, 0.05, 0.6))
    data3[i] = data3[i] * (Gaussian_filter(Grid.zcgrid[i], 1.75, 0.05, 0.5))
    data4[i] = data3[i] * (Gaussian_filter(Grid.zcgrid[i], 1.75, 0.05,0.7))

halo0 = go.Figure(data=go.Volume(
    x=z.flatten(),
    y=ra.flatten(),
    z=dec.flatten(),
    value=data3.flatten(),
    showscale=False,
    isomin=0.01,
    isomax=0.5,
    opacity=0.2,  # needs to be small to see through all surfaces
    surface_count=21,  # needs to be a large number for good volume rendering
    colorscale='Blues'))

halo1=go.Volume(
    x=z.flatten(),
    y=ra.flatten()+2,
    z=dec.flatten()+2,
    value=data2.flatten(),
    showscale=False,
    isomin=0.01,
    isomax=0.5,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=21, # needs to be a large number for good volume rendering
    colorscale='Blues')

halo2=go.Volume(
    x=z.flatten(),
    y=ra.flatten()-5,
    z=dec.flatten()-3,
    value=data3.flatten(),
    showscale=False,
    isomin=0.01,
    isomax=0.5,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=21, # needs to be a large number for good volume rendering
    colorscale='Blues')

halo3=go.Volume(
    x=z.flatten(),
    y=ra.flatten()+1,
    z=dec.flatten()+1,
    value=data4.flatten(),
    showscale=False,
    isomin=0.01,
    isomax=0.5,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=21, # needs to be a large number for good volume rendering
    colorscale='Blues')

fig = go.Figure(data = [halo1,halo2,halo3])

fig.update_layout(scene=dict(
    xaxis=dict(range=[-0.5,3],
        backgroundcolor="rgba(0, 0, 0, 0)",
        gridcolor="lightslategrey",
        showbackground=False,
        zerolinecolor="white", nticks=5, tickvals=[0.25, 0.75, 1.25, 1.75, 2.25],
        ticktext=['0.25', '0.75', '1.25', '1.75', '2.25']),
    yaxis=dict(range=[-12,12],
        backgroundcolor="rgba(0, 0, 0, 0)",
        gridcolor="lightslategrey",
        showbackground=False,
        zerolinecolor="white", nticks=4, tickvals=[-7.5, -2.5, 2.5, 7.5], ticktext=['-15', '-5', '5', '15']),
    zaxis=dict(range=[-12,12],
        backgroundcolor="rgba(0, 0, 0, 0)",
        gridcolor="lightslategrey",
        showbackground=False,
        zerolinecolor="white", nticks=4, tickvals=[-7.5, -2.5, 2.5, 7.5], ticktext=['-15', '-5', '5', '15'])),
    xaxis_title="Redshift",
    yaxis_title="ra",
)
fig.update_layout(scene=dict(
    xaxis_title='Redshift',
    yaxis_title='ra (arcmin)',
    zaxis_title='dec (arcmin)'), font=dict(size=15, ))
fig.show()
# , tickvals = [-7.5, 2.5, 2.5,7.5], ticktext = ['-15', '-5', '5', '15']
