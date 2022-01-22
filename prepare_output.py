#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script processes output from Beaufort Gyre Aronnax experiments
(https://github.com/edoddridge/aronnax)
@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from scipy.interpolate import RegularGridInterpolator
import aronnax.driver as drv
from aronnax.utils import working_directory
from aronnax.core import interpret_raw_file, Grid
from netCDF4 import Dataset
from glob import glob
from noise import pnoise2
from tqdm import tqdm

def gen_zeta(u, v, nl):
    zeta = np.zeros((nl, np.shape(v)[1], np.shape(u)[2]))
    for l in range(nl):
        u_ext = np.pad(u, ((0, 0), (1, 1), (0, 0)), mode='constant', constant_values=0)
        v_ext = np.pad(v, ((0, 0), (0, 0), (1, 1)), mode='constant', constant_values=0)
        zeta[l, :, :] = (((v_ext[l, :, 1:] - v_ext[l, :, :-1])/param['res']) -
                         ((u_ext[l, 1:, :] - u_ext[l, :-1, :])/param['res']))

    return zeta

def zeta2h(zeta):
    zeta_mod = zeta[:, :-1, :-1] + zeta[:, 1:, :-1] + zeta[:, :-1, 1:] + zeta[:, 1:, 1:]
    return zeta_mod/4

def gen_vel(u, v):
    u_mod = 0.5*(u[:, :, :-1] + u[:, :, 1:])
    v_mod = 0.5*(v[:, :-1, :] + v[:, 1:, :])

    return np.sqrt(u_mod**2 + v_mod**2)

def gen_norm(u, v, u_norm, v_norm):
    u_n = u*u_norm
    v_n = v*v_norm
    return 0.5*(u_n[:, :, 1:]+u_n[:, :, :-1])+0.5*(v_n[:, 1:, :]+v_n[:, :-1, :])

##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

slope_width = 0 # km

dirs = {'root'  : os.path.dirname(os.path.realpath(__file__)) + '/'}
dirs['linear3'] = dirs['root'] + 'linear3'
dirs['sim'] = dirs['linear3'] + '/' + str(slope_width)
dirs['output'] = dirs['sim'] + '/output/'

param = {# DOMAIN GEOMETRY
         'res'       :    5e3,     # Resolution in metres
         'Lx'        : 1510e3,     # Domain width in metres
         'Ly'        : 2010e3,     # Domain height in metres
         'GR'        :  750e3,     # Gyre radius in metres
         'GCx'       :  755e3,     # Gyre centre (x coordinate)
         'GCy'       : 1255e3,     # Gyre centre (y coordinate)
         'CRx'       :  250e3,     # Channel radius in metres

         'Lz_min'    :  0.5e3,     # Maximum domain depth in metres
         'Lz_max'    :  4.0e3,     # Maximum domain depth in metres
         'H1'        :  0.8e2,     # Layer 1 depth in metres
         'H2'        :  0.8e2,     # Layer 2 depth in metres
         'noise'     :   True,     # Add noise to bathymetry

         'Sx'        :  slope_width*1e3,     # Slope width in metres

         'layers'    :      3,     # Number of layers

         # WIND FORCING
         'tau_max'   :2.0e-2,     # Maximum wind stress (N/m^2)

         # SPONGE REGION
         'sponge_ts' :   1e-1,     # Sponge timescale (days)
         'grad_Ly'   :  300e3,     # Distance over which sponging is downramped

         # CORIOLIS
         'lat'       :    90.,     # Latitude (deg) to calculate f at

         # NUMERICS
         'dt'        :   100.,     # Time step (seconds)
         'sim_time'  :  9000.,     # Simulation runtime (days)
         'snap_freq' :     1.,     # Output frequency (days)
         'av_freq'   :   360.,     # Average frequency (days)
         'chk_freq'  :   360.,     # Checkpoint frequency (days)

         # PARAMETERS FOR AZIMUTHAL MEAN
         'n_rad'     :     50,     # Radial samples
         'n_azi'     :    360,     # Azimuthal samples
         }

# Add some extra calculated variables
param['nx'] = int(param['Lx']/param['res'])
param['ny'] = int(param['Ly']/param['res'])
param['f0'] = np.sin(param['lat']*np.pi/180)*4*np.pi/(86400)

fh = {'av'  : {'h'   : sorted(glob(dirs['output'] + 'av.h.*')),
               'eta' : sorted(glob(dirs['output'] + 'av.eta.*')),
               'u'   : sorted(glob(dirs['output'] + 'av.u.*')),
               'v'   : sorted(glob(dirs['output'] + 'av.v.*')),},
      'inst': {'h'   : sorted(glob(dirs['output'] + 'snap.h.*')),
               'eta' : sorted(glob(dirs['output'] + 'snap.eta.*')),
               'u'   : sorted(glob(dirs['output'] + 'snap.u.*')),
               'v'   : sorted(glob(dirs['output'] + 'snap.v.*')),},
      'out' : dirs['sim'] + '/processed_output_' + str(slope_width) + '.nc'}

##############################################################################
# READ DATA                                                                  #
##############################################################################



# Firstly prepare grids and axes
nt = len(fh['inst']['h'])
nav = len(fh['av']['h'])
time = (1+np.arange(nt))*param['snap_freq']/360 # Time axis in years

model_grid = Grid(nx=param['nx'], ny=param['ny'], layers=param['layers'],
                  dx=param['res'], dy=param['res'])

shape = {'h'   : (nt, model_grid.layers, len(model_grid.y), len(model_grid.x)),
         'eta' : (nt, model_grid.layers, len(model_grid.y), len(model_grid.x)),
         'zeta': (nt, model_grid.layers, len(model_grid.yp1), len(model_grid.xp1)),
         'u'   : (nt, model_grid.layers, len(model_grid.y), len(model_grid.xp1)),
         'v'   : (nt, model_grid.layers, len(model_grid.yp1), len(model_grid.x)),}

# Calculate the average fields
h_av = np.zeros(shape['h'][1:], dtype=np.float32)
eta_av = np.zeros(shape['eta'][2:], dtype=np.float32)
u_av = np.zeros(shape['u'][1:], dtype=np.float32)
v_av = np.zeros(shape['v'][1:], dtype=np.float32)
zeta_av = np.zeros(shape['zeta'][1:], dtype=np.float32)

for i, fhi in tqdm(enumerate(zip(fh['av']['h'], fh['av']['eta'], fh['av']['u'], fh['av']['v'])),
                   total=len(fh['av']['h'])):
    h = interpret_raw_file(fhi[0], param['nx'], param['ny'], param['layers'])
    eta = interpret_raw_file(fhi[1], param['nx'], param['ny'], param['layers'])[0, :, :]
    u = interpret_raw_file(fhi[2], param['nx'], param['ny'], param['layers'])
    v = interpret_raw_file(fhi[3], param['nx'], param['ny'], param['layers'])

    zeta = gen_zeta(u, v, param['layers'])

    h_av += h
    eta_av += eta
    u_av += u
    v_av += v
    zeta_av += zeta

h_av /= nav
eta_av /= nav
u_av /= nav
v_av /= nav
zeta_av /= nav

# Calculate azimuthal time mean quantities
momentum_flux_atm = np.zeros((param['layers'], param['n_rad']), dtype=np.float32)
momentum_flux_convergence_atm = np.zeros((param['layers'], param['n_rad']), dtype=np.float32)
bolus_flux_atm = np.zeros((param['layers'], param['n_rad']), dtype=np.float32)
pv_atm = np.zeros((param['layers'], param['n_rad']), dtype=np.float32)
layer_thickness_atm = np.zeros((param['layers'], param['n_rad']), dtype=np.float32)
eta_atm = np.zeros((1, param['n_rad']), dtype=np.float32)
vorticity_atm = np.zeros((param['layers'], param['n_rad']), dtype=np.float32)
azimuthal_velocity_atm = np.zeros((param['layers'], param['n_rad']), dtype=np.float32)

# Calculate polar coordinate grids
X, Y = np.meshgrid(model_grid.x, model_grid.y)
Xu, Yu = np.meshgrid(model_grid.xp1, model_grid.y)
Xv, Yv = np.meshgrid(model_grid.x, model_grid.yp1)

Xu = Xu - param['GCx']
Yu = Yu - param['GCy']
Xv = Xv - param['GCx']
Yv = Yv - param['GCy']

ru     = np.sqrt(Xu**2 + Yu**2)
rv     = np.sqrt(Xv**2 + Yv**2)

radnorm_u = Xu/ru # Normal radial vector, u component
radnorm_v = Yv/rv # Normal radial vector, v component

azinorm_u = -Yu/ru
azinorm_v = Xv/rv

aziu = np.pi-np.fliplr(np.arctan2(Yu, Xu))
aziv = np.pi-np.fliplr(np.arctan2(Yv, Xv))

# Generate interpolation points
r_axis = np.linspace(0, param['GR'], num=param['n_rad']+1)[1:]
azi_axis = np.linspace(0, 2*np.pi, num=param['n_azi']+1)[1:]

input_space = np.meshgrid(r_axis, azi_axis)

x_sample = param['GCx'] + input_space[0]*np.cos(input_space[1])
x_sample = x_sample.flatten()
y_sample = param['GCy'] + input_space[0]*np.sin(input_space[1])
y_sample = y_sample.flatten()

# Carry out interpolations for vort/eta/thickness
for l in range(param['layers']):
    h_int = RegularGridInterpolator((model_grid.y, model_grid.x), h_av[l, :, :], method='linear')
    z_int = RegularGridInterpolator((model_grid.yp1, model_grid.xp1), zeta_av[l, :, :], method='linear')
    vel_azi_int = RegularGridInterpolator((model_grid.y, model_grid.x), gen_norm(u_av, v_av, azinorm_u, azinorm_v)[l, :, :], method='linear')

    layer_thickness_atm[l, :] = np.mean(h_int(np.array([y_sample, x_sample]).T).reshape((param['n_azi'], -1)), axis=0)
    vorticity_atm[l, :] = np.mean(z_int(np.array([y_sample, x_sample]).T).reshape((param['n_azi']), -1), axis=0)
    azimuthal_velocity_atm[l, :] = np.mean(vel_azi_int(np.array([y_sample, x_sample]).T).reshape((param['n_azi']), -1), axis=0)

eta_int = RegularGridInterpolator((model_grid.y, model_grid.x), eta_av[:, :], method='linear')
eta_atm[0, :] = np.mean(eta_int(np.array([y_sample, x_sample]).T).reshape((param['n_azi'], -1)), axis=0)

for i, fhi in tqdm(enumerate(zip(fh['inst']['h'], fh['inst']['eta'], fh['inst']['u'], fh['inst']['v'])),
                   total=len(fh['inst']['h'])):

    h   = interpret_raw_file(fhi[0], param['nx'], param['ny'], param['layers'])
    eta = interpret_raw_file(fhi[1], param['nx'], param['ny'], param['layers'])[0, :, :]
    u   = interpret_raw_file(fhi[2], param['nx'], param['ny'], param['layers'])
    v   = interpret_raw_file(fhi[3], param['nx'], param['ny'], param['layers'])

    h_dash   = h - h_av
    eta_dash = eta - eta_av
    u_dash   = u - u_av
    v_dash   = v - v_av

    zeta = gen_zeta(u, v, param['layers'])
    zeta_dash = zeta - zeta_av

    pv = (zeta2h(zeta) + param['f0'])/h

    # Now calculate the radial component of flow
    vel_rad = gen_norm(u, v, radnorm_u, radnorm_v) # Radial velocity    (outward = +ve)
    vel_azi = gen_norm(u, v, azinorm_u, azinorm_v) # Azimuthal velocity (anticlockwise = +ve)

    vel_rad_dash = gen_norm(u_dash, v_dash, radnorm_u, radnorm_v) # Radial velocity difference    (outward = +ve)
    vel_azi_dash = gen_norm(u_dash, v_dash, azinorm_u, azinorm_v) # Azimuthal velocity difference (anticlockwise = +ve)

    # vel = gen_vel(u, v) # Absolute velocity

    # Now interpolate
    for l in range(param['layers']):
        momentum_flux_int = RegularGridInterpolator((model_grid.y, model_grid.x), vel_rad_dash[l, :, :]*vel_azi_dash[l, :, :], method='linear')
        momentum_flux_convergence_int = RegularGridInterpolator((model_grid.y, model_grid.x), vel_rad_dash[l, :, :]*zeta2h(zeta_dash)[l, :, :], method='linear')
        bolus_flux_int = RegularGridInterpolator((model_grid.y, model_grid.x), vel_rad_dash[l, :, :]*h_dash[l, :, :]*param['f0']/h[l, :, :], method='linear')
        pv_int = RegularGridInterpolator((model_grid.y, model_grid.x), pv[l, :, :], method='linear')

        momentum_flux_atm[l, :] = np.mean(momentum_flux_int(np.array([y_sample, x_sample]).T).reshape((param['n_azi'], -1)), axis=0)
        momentum_flux_convergence_atm[l, :] = np.mean(momentum_flux_convergence_int(np.array([y_sample, x_sample]).T).reshape((param['n_azi'], -1)), axis=0)
        bolus_flux_atm[l, :] = np.mean(bolus_flux_int(np.array([y_sample, x_sample]).T).reshape((param['n_azi'], -1)), axis=0)
        pv_atm[l, :] = np.mean(pv_int(np.array([y_sample, x_sample]).T).reshape((param['n_azi'], -1)), axis=0)


    # f, ax = plt.subplots(1, 1, figsize=(10, 13), constrained_layout=True)
    # ax.pcolormesh(model_grid.x, model_grid.y, vel[0,:,:], vmin=0, vmax=1, cmap=cmr.voltage_r)
    # ax.set_aspect('equal')



    # Now interpolate values











# Create netcdf file
with Dataset(fh['out'], mode='w') as nc:
    nc.createDimension('time', nt)
    nc.createDimension('layers', param['layers'])
    nc.createDimension('sfc', 1)
    nc.createDimension('x', len(model_grid.x))
    nc.createDimension('xp1', len(model_grid.xp1))
    nc.createDimension('y', len(model_grid.y))
    nc.createDimension('yp1', len(model_grid.yp1))

    # Create the axes
    nc.createVariable('x', 'i4', ('x'), zlib=True)
    nc.variables['x'].long_name = 'x_coordinate'
    nc.variables['x'].units = 'metres'
    nc.variables['x'].standard_name = 'x_coordinate'
    nc.variables['x'][:] = model_grid.x

    nc.createVariable('xp1', 'i4', ('xp1'), zlib=True)
    nc.variables['xp1'].long_name = 'xp1_coordinate'
    nc.variables['xp1'].units = 'metres'
    nc.variables['xp1'].standard_name = 'xp1_coordinate'
    nc.variables['xp1'][:] = model_grid.xp1

    nc.createVariable('y', 'i4', ('y'), zlib=True)
    nc.variables['y'].long_name = 'y_coordinate'
    nc.variables['y'].units = 'metres'
    nc.variables['y'].standard_name = 'y_coordinate'
    nc.variables['y'][:] = model_grid.y

    nc.createVariable('yp1', 'i4', ('yp1'), zlib=True)
    nc.variables['yp1'].long_name = 'x_coordinate'
    nc.variables['yp1'].units = 'metres'
    nc.variables['yp1'].standard_name = 'yp1_coordinate'
    nc.variables['yp1'][:] = model_grid.yp1

    # Create the variables
    nc.createVariable('time', 'f4', ('time'), zlib=True)
    nc.variables['time'].long_name = 'years_since_equilibrated_run_start'
    nc.variables['time'].units = 'second'
    nc.variables['time'].standard_name = 'time'
    nc.variables['time'].axis = 'T'

    nc.createVariable('u_inst', 'f4', ('time', 'layers', 'y', 'xp1'), zlib=True)
    nc.variables['u_inst'].long_name = 'instantaneous_u_velocity_at_u_points'
    nc.variables['u_inst'].units = 'metres per second'
    nc.variables['u_inst'].standard_name = 'u_inst'
    nc.variables['u_inst'].coordinates = 'y xp1'

    nc.createVariable('v_inst', 'f4', ('time', 'layers', 'yp1', 'x'), zlib=True)
    nc.variables['v_inst'].long_name = 'instantaneous_v_velocity_at_v_points'
    nc.variables['v_inst'].units = 'metres per second'
    nc.variables['v_inst'].standard_name = 'v_inst'
    nc.variables['v_inst'].coordinates = 'yp1 x'

    nc.createVariable('h_inst', 'f4', ('time', 'layers', 'y', 'x'), zlib=True)
    nc.variables['h_inst'].long_name = 'instantaneous_layer_thickness_at_tracer_points'
    nc.variables['h_inst'].units = 'metres'
    nc.variables['h_inst'].standard_name = 'h_inst'
    nc.variables['h_inst'].coordinates = 'y x'

    nc.createVariable('eta_inst', 'f4', ('time', 'sfc', 'y', 'x'), zlib=True)
    nc.variables['eta_inst'].long_name = 'instantaneous_free_surface_at_tracer_points'
    nc.variables['eta_inst'].units = 'metres'
    nc.variables['eta_inst'].standard_name = 'eta_inst'
    nc.variables['eta_inst'].coordinates = 'y x'

    nc.createVariable('zeta_inst', 'f4', ('time', 'layers', 'y', 'x'), zlib=True)
    nc.variables['zeta_inst'].long_name = 'instantaneous_vorticity_at_tracer_points'
    nc.variables['zeta_inst'].units = 'per second'
    nc.variables['zeta_inst'].standard_name = 'zeta_inst'
    nc.variables['zeta_inst'].coordinates = 'y x'

    nc.createVariable('u_av', 'f4', ('layers', 'y', 'xp1'), zlib=True)
    nc.variables['u_av'].long_name = 'average_u_velocity_at_u_points'
    nc.variables['u_av'].units = 'metres per second'
    nc.variables['u_av'].standard_name = 'u_av'
    nc.variables['u_av'].coordinates = 'y xp1'

    nc.createVariable('v_av', 'f4', ('layers', 'yp1', 'x'), zlib=True)
    nc.variables['v_av'].long_name = 'average_v_velocity_at_v_points'
    nc.variables['v_av'].units = 'metres per second'
    nc.variables['v_av'].standard_name = 'v_iav'
    nc.variables['v_av'].coordinates = 'yp1 x'

    nc.createVariable('h_av', 'f4', ('layers', 'y', 'x'), zlib=True)
    nc.variables['h_av'].long_name = 'average_layer_thickness_at_tracer_points'
    nc.variables['h_av'].units = 'metres'
    nc.variables['h_av'].standard_name = 'h_av'
    nc.variables['h_av'].coordinates = 'y x'

    nc.createVariable('eta_av', 'f4', ('sfc', 'y', 'x'), zlib=True)
    nc.variables['eta_av'].long_name = 'average_free_surface_at_tracer_points'
    nc.variables['eta_av'].units = 'metres'
    nc.variables['eta_av'].standard_name = 'eta_av'
    nc.variables['eta_av'].coordinates = 'y x'

    nc.createVariable('zeta_av', 'f4', ('layers', 'y', 'x'), zlib=True)
    nc.variables['zeta_av'].long_name = 'average_vorticity_at_tracer_points'
    nc.variables['zeta_av'].units = 'per second'
    nc.variables['zeta_av'].standard_name = 'zeta_av'
    nc.variables['zeta_av'].coordinates = 'y x'

    nc.variables['time_inst'][:] = time





# for i, fhi in tqdm(enumerate(fh['inst']['h']), total=len(fh['inst']['h'])):
#     data = interpret_raw_file(fh, param['nx'], param['ny'], param['layers'])
#     h_ts[i] = data[0, int(param['ny']-((param['nx']+1)/2)), int((param['nx']+1)/2)]

