#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script runs the idealised Beaufort Gyre experiments in Aronnax
(https://github.com/edoddridge/aronnax)
@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cm
import aronnax.driver as drv
from aronnax.utils import working_directory
from aronnax.core import interpret_raw_file, Grid
from netCDF4 import Dataset
from glob import glob
from noise import pnoise2

##############################################################################
# WIND STRESS FORMULATION                                                    #
##############################################################################
'''
DAVIS ET AL (2014) (TRIG) FORMULATION
From Davis et al. (2014), we use a radially symmetric wind stress field with
curl that is maximum at the centre of the gyre and zero at the gyre boundary.
The wind stress in the azimuthal direction (tau_phi) is:

    tau_phi = (a/(br))*int[(br)*cos^2(br) dr]
            = (a/(br))*[(1/4)*(br^2) + (r/4)*sin(2br) + (1/(8b))*cos(2br)]
            = ((ar)/4) + (a/(4b))*sin(2br) + (a/(8(b^2)r))*cos(2br)

The x and y components of wind stress are:

    tau_x =  sin(theta) * tau_phi
    tau_y = -cos(theta) * tau_phi

The curl in the z direction is:

    curl_z(tau) = a*cos^2(br)

To satisfy curl_z(tau)[r=R] = 0, b=pi/(2R) where R is the gyre radius. The
constant a linearly scales the wind stress curl.

For the outside of the gyre, we set the wind stress curl to
the following zero-curl field:

    tau_x = c(y/r^2)
    tau_y = c(-x/r^2)

Where c = (1/4)*(aR^2) + (1/(32pi^2))*(aR) for continuity with the gyre winds


POLYNOMIAL FORMULATION (DEFAULT)
To avoid the wind stress being unbounded in the centre of the domain, we can
alternatively use a polynomial formulation for the wind stress (also with a
curl that is maximum at the centre of the gyre and zero at the boundary).
The wind stress in the azimuthal direction (tau_phi) is:

    tau_phi = (a/r)*int[(r^2/2) - (b^2r^4/4)]
            = (ar/4)*(2-b^2r^2)

The x and y components of wind stress are:

    tau_x =  sin(theta) * tau_phi
    tau_y = -cos(theta) * tau_phi

The curl in the z direction is:

    curl_z(tau) = a(1-b^2r^2)

To satisfy curl_z(tau)[r=R] = 0, b=1/R where R is the gyre radius. We can
define a in terms of a defined maximum wind stress tau_max. The maximum wind
stress occurs at r = dR where d = sqrt(2/3), therefore:

    tau_max = (adR/4)*(2-b^2d^2R^2)
    a = (4*tau_max)/(dR*(2-b^2d^2R^2))

As in Davis et al. (2014), outside of the gyre, we set the wind stress curl to
the following zero-curl field:

    tau_x = c(y/r^2)
    tau_y = c(-x/r^2)

Where c = (aR^2/4)*(2-b^2r^2) for continuity with the gyre winds

'''
##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

slope_width = 0  # km

dirs = {'root'  : os.path.dirname(os.path.realpath(__file__)) + '/'}
dirs['ctrl']   = dirs['root'] + 'ctrl'
dirs['linear3'] = dirs['root'] + 'linear3'
dirs['sim'] = dirs['linear3'] + '/' + str(slope_width)

param = {# DOMAIN GEOMETRY
         'res'       :    5e3,     # Resolution in metres
         'Lx'        : 1510e3,     # Domain width in metres
         'Ly'        : 2010e3,     # Domain height in metres
         'GR'        :  750e3,     # Gyre radius in metres
         'GCx'       :  755e3,     # Gyre centre (x coordinate)
         'GCy'       : 1255e3,     # Gyre centre (y coordinate)
         'CRx'       :  200e3,     # Channel radius in metres

         'Lz_min'    :  0.5e3,     # Maximum domain depth in metres
         'Lz_max'    :  4.0e3,     # Maximum domain depth in metres
         'H1'        :  1.0e2,     # Layer 1 depth in metres
         'H2'        :  1.0e2,     # Layer 2 depth in metres
         'noise'     :   True,     # Add noise to bathymetry

         'Sx'        :  slope_width*1e3,     # Slope width in metres

         'layers'    :      3,     # Number of layers

         # WIND FORCING
         'tau_max'   :1.75e-1,     # Maximum wind stress (N/m^2)

         # SPONGE REGION
         'sponge_ts' :    1e0,     # Sponge timescale (days)
         'grad_Ly'   :  300e3,     # Distance over which sponging is downramped

         # CORIOLIS
         'lat'       :    90.,     # Latitude (deg) to calculate f at

         # NUMERICS
         'dt'        :   120.,     # Time step (seconds)
         'sim_time'  :  7200.,     # Simulation runtime (days)
         'snap_freq' :     1.,     # Output frequency (days)
         'av_freq'   :    30.,     # Average frequency (days)
         'chk_freq'  :   360.,     # Checkpoint frequency (days)

         # FILE NAMES
         'netcdf_name'  : str(slope_width) + '.nc'
         }

# Add some extra calculated variables
param['nx'] = int(param['Lx']/param['res'])
param['ny'] = int(param['Ly']/param['res'])

##############################################################################
# FUNCTIONS                                                                  #
##############################################################################

def convert_to_netcdf(dir_in, dir_out, **kwargs):
    # This script converts the output from Aronnax from binary to netcdf
    # format

    # Check if animations directory exists
    if not os.path.isdir(dirs['sim'] + '/netcdf-output'):
        os.makedirs(dirs['sim'] + '/netcdf-output')

    # dir_in : where the raw binary files from Aronnax are found
    # dir_out: where the processed netcdf files should be placed

    proc_snap, proc_av = False, False

    if {'snap'} <= kwargs.keys():
        if kwargs['snap']:
            proc_snap = True
            snap_files = {}

    if {'av'} <= kwargs.keys():
        if kwargs['av']:
            proc_av = True
            av_files = {}

    if (proc_snap == False) and (proc_av == False):
        raise NotImplementedError('Must select at least one of snap or av!')

    # Retrieve file names
    var_list = ['h', 'eta', 'u', 'v']

    if proc_snap:
        for var in var_list:
            snap_files[var] = sorted(glob(dir_in + 'snap.' + var + '.*'))

        nt_snap = int(param['sim_time']/param['snap_freq'] + 1)

    if proc_av:
        for var in var_list:
            av_files[var] = sorted(glob(dir_in + 'av.' + var + '.*'))

        nt_av = int(param['sim_time']/param['av_freq'])

    # Set up the model grid
    model_grid = Grid(nx=param['nx'], ny=param['ny'], layers=param['layers'],
                      dx=param['res'], dy=param['res'])


    if not (len(snap_files['h']) == nt_snap)*(len(av_files['h']) == nt_av):
        raise ValueError('Expected ' + str(nt_snap) + ' snapshots but found ' +
                         str(len(snap_files['h'])) + '. Expected ' + str(nt_av) +
                         ' averages but found ' + str(len(av_files['h'])) + '.')

    # Now read in the files
    if proc_snap:
        # Generate empty container arrays
        snap = {}
        for var in var_list:
            var_shape = interpret_raw_file(snap_files[var][0],
                                           param['nx'],
                                           param['ny'],
                                           param['layers']).shape
            snap[var] = np.zeros(((nt_snap,) + var_shape))

        # Fill array
        for var in ['h', 'eta', 'u', 'v']:
            for i, fh in enumerate(snap_files[var]):
                snap[var][i, :, :, :] = interpret_raw_file(fh,
                                                           param['nx'],
                                                           param['ny'],
                                                           param['layers'])

    if proc_av:
        # Generate empty container arrays
        av = {}
        for var in var_list:
            var_shape = interpret_raw_file(av_files[var][0],
                                           param['nx'],
                                           param['ny'],
                                           param['layers']).shape
            av[var] = np.zeros(((nt_av,) + var_shape))

        # Fill array
        for var in ['h', 'eta', 'u', 'v']:
            for i, fh in enumerate(av_files[var]):
                av[var][i, :, :, :] = interpret_raw_file(fh,
                                                         param['nx'],
                                                         param['ny'],
                                                         param['layers'])
    # Now generate the time array
    if proc_snap:
        snap_time = np.arange(0, (param['sim_time'] + param['snap_freq'])*86400,
                              param['snap_freq']*86400)

    if proc_av:
        av_time = np.arange(param['av_freq']*86400,
                              (param['sim_time'] + param['av_freq'])*86400,
                              param['av_freq']*86400)

    # Now export to netcdf
    with Dataset(dirs['sim'] + '/netcdf-output/' + param['netcdf_name'],
                 mode='w') as nc:

        # Create the dimensions
        if proc_snap:
            nc.createDimension('time_snap', nt_snap)

        if proc_av:
            nc.createDimension('time_av', nt_av)

        nc.createDimension('layers', param['layers'])
        nc.createDimension('sfc', 1)
        nc.createDimension('x', len(model_grid.x))
        nc.createDimension('xp1', len(model_grid.xp1))
        nc.createDimension('y', len(model_grid.y))
        nc.createDimension('yp1', len(model_grid.yp1))

        # Create the variables

        nc.createVariable('x', 'f8', ('x'), zlib=True)
        nc.variables['x'].long_name = 'x_coordinate'
        nc.variables['x'].units = 'metres'
        nc.variables['x'].standard_name = 'x_coordinate'
        nc.variables['x'][:] = model_grid.x

        nc.createVariable('xp1', 'f8', ('xp1'), zlib=True)
        nc.variables['xp1'].long_name = 'xp1_coordinate'
        nc.variables['xp1'].units = 'metres'
        nc.variables['xp1'].standard_name = 'xp1_coordinate'
        nc.variables['xp1'][:] = model_grid.xp1

        nc.createVariable('y', 'f8', ('y'), zlib=True)
        nc.variables['y'].long_name = 'y_coordinate'
        nc.variables['y'].units = 'metres'
        nc.variables['y'].standard_name = 'y_coordinate'
        nc.variables['y'][:] = model_grid.y

        nc.createVariable('yp1', 'f8', ('yp1'), zlib=True)
        nc.variables['yp1'].long_name = 'x_coordinate'
        nc.variables['yp1'].units = 'metres'
        nc.variables['yp1'].standard_name = 'yp1_coordinate'
        nc.variables['yp1'][:] = model_grid.yp1

        if proc_snap:
            nc.createVariable('time_snap', 'f8', ('time_snap'), zlib=True)
            nc.variables['time_snap'].long_name = 'seconds_since_initialization_(snapshots)'
            nc.variables['time_snap'].units = 'second'
            nc.variables['time_snap'].standard_name = 'snapshot_time'
            nc.variables['time_snap'].axis = 'T'

            nc.createVariable('u_snap', 'f8', ('time_snap', 'layers', 'y', 'xp1'), zlib=True)
            nc.variables['u_snap'].long_name = 'u_velocity_snapshot_at_u_points'
            nc.variables['u_snap'].units = 'metres per second'
            nc.variables['u_snap'].standard_name = 'u_snap'
            nc.variables['u_snap'].coordinates = 'y xp1'

            nc.createVariable('v_snap', 'f8', ('time_snap', 'layers', 'yp1', 'x'), zlib=True)
            nc.variables['v_snap'].long_name = 'v_velocity_snapshot_at_v_points'
            nc.variables['v_snap'].units = 'metres per second'
            nc.variables['v_snap'].standard_name = 'v_snap'
            nc.variables['v_snap'].coordinates = 'yp1 x'

            nc.createVariable('h_snap', 'f8', ('time_snap', 'layers', 'y', 'x'), zlib=True)
            nc.variables['h_snap'].long_name = 'layer_thickness_snapshot_at_tracer_points'
            nc.variables['h_snap'].units = 'metres'
            nc.variables['h_snap'].standard_name = 'h_snap'
            nc.variables['h_snap'].coordinates = 'y x'

            nc.createVariable('eta_snap', 'f8', ('time_snap', 'sfc', 'y', 'x'), zlib=True)
            nc.variables['eta_snap'].long_name = 'free_surface_snapshot_at_tracer_points'
            nc.variables['eta_snap'].units = 'metres'
            nc.variables['eta_snap'].standard_name = 'eta_snap'
            nc.variables['eta_snap'].coordinates = 'y x'

            nc.variables['time_snap'][:] = snap_time
            nc.variables['u_snap'][:] = snap['u']
            nc.variables['v_snap'][:] = snap['v']
            nc.variables['h_snap'][:] = snap['h']
            nc.variables['eta_snap'][:] = snap['eta']

        if proc_av:
            nc.createVariable('time_av', 'f8', ('time_av'), zlib=True)
            nc.variables['time_av'].long_name = 'seconds_since_initialization_at_end_of_averaging_period_(averages)'
            nc.variables['time_av'].units = 'second'
            nc.variables['time_av'].standard_name = 'average_time'
            nc.variables['time_av'].axis = 'T'
            nc.variables['time_av'].averaging_period = str(param['av_freq']) + ' days'

            nc.createVariable('u_av', 'f8', ('time_av', 'layers', 'y', 'xp1'), zlib=True)
            nc.variables['u_av'].long_name = 'u_velocity_average_at_u_points'
            nc.variables['u_av'].units = 'metres per second'
            nc.variables['u_av'].standard_name = 'u_av'
            nc.variables['u_av'].coordinates = 'y xp1'

            nc.createVariable('v_av', 'f8', ('time_av', 'layers', 'yp1', 'x'), zlib=True)
            nc.variables['v_av'].long_name = 'v_velocity_average_at_v_points'
            nc.variables['v_av'].units = 'metres per second'
            nc.variables['v_av'].standard_name = 'v_av'
            nc.variables['v_av'].coordinates = 'yp1 x'

            nc.createVariable('h_av', 'f8', ('time_av', 'layers', 'y', 'x'), zlib=True)
            nc.variables['h_av'].long_name = 'layer_thickness_average_at_tracer_points'
            nc.variables['h_av'].units = 'metres'
            nc.variables['h_av'].standard_name = 'h_av'
            nc.variables['h_av'].coordinates = 'y x'

            nc.createVariable('eta_av', 'f8', ('time_av', 'sfc', 'y', 'x'), zlib=True)
            nc.variables['eta_av'].long_name = 'free_surface_average_at_tracer_points'
            nc.variables['eta_av'].units = 'metres'
            nc.variables['eta_av'].standard_name = 'eta_av'
            nc.variables['eta_av'].coordinates = 'y x'

            nc.variables['time_av'][:] = av_time
            nc.variables['u_av'][:] = av['u']
            nc.variables['v_av'][:] = av['v']
            nc.variables['h_av'][:] = av['h']
            nc.variables['eta_av'][:] = av['eta']

        # Save parameters to netcdf
        nc.res = param['res']
        nc.Lx = param['Lx']
        nc.Ly = param['Ly']
        nc.GR = param['GR']
        nc.GCx = param['GCx']
        nc.GCy = param['GCy']
        nc.CRx = param['CRx']

        nc.Lz_min = param['Lz_min']
        nc.Lz_max = param['Lz_max']
        nc.H1     = param['H1']
        nc.H2     = param['H2']
        nc.Sx     = param['Sx']
        nc.layers = param['layers']

        nc.sponge_ts = param['sponge_ts']

        nc.lat = param['lat']

        nc.tau_max = param['tau_max']

        nc.dt = param['dt']
        nc.sim_time = param['sim_time']
        nc.snap_freq = param['snap_freq']
        nc.av_freq = param['av_freq']
        nc.chk_freq = param['chk_freq']


def run_beaufort_linear3():

    ###########################################################################
    # SET UP INPUTS                                                           #
    ###########################################################################

    def plot_field(X, Y, field, cmap, fn, **kwargs):
        # Mask with the lsm
        if {'mask'} <= kwargs.keys():
            if kwargs['mask']:
                field = np.ma.masked_array(field, mask=1-lsm(X, Y, plot=False))

        # Plot a field
        f, ax = plt.subplots(1, 1, figsize=(7,6))

        if {'vmin', 'vmax'} <= kwargs.keys():
            img   = ax.pcolormesh(X/1e3, Y/1e3, field, cmap=cmap,
                                  vmin=kwargs['vmin'],
                                  vmax=kwargs['vmax'])
        else:
            img   = ax.pcolormesh(X/1e3, Y/1e3, field, cmap=cmap)


        ax.set_aspect('equal')
        ax.set_xlabel('x coordinate (km)')
        ax.set_ylabel('y coordinate (km)')
        plt.colorbar(img)
        plt.savefig(dirs['sim'] + '/figures/' + fn)
        plt.close()

    def gen_bath(X, Y):
        # Centre the coordinates around the gyre centre
        Xp = X - param['GCx']
        Yp = Y - param['GCy']
        r  = np.sqrt(Xp**2 + Yp**2)

        if param['Sx'] == 0:
            bath = np.ones_like(X)*param['Lz_max']
        else:
            Sz = param['Lz_min'] - param['Lz_max']
            bath = (Sz*r/param['Sx']) + param['Lz_min'] - (param['GR']*Sz/param['Sx'])
            bath = np.minimum(param['Lz_max']*np.ones_like(bath), bath)
            bath = np.maximum(param['Lz_min']*np.ones_like(bath), bath)

        if param['noise']:
            noise = np.zeros_like(bath)

            freq = 16
            for y in range(bath.shape[0]):
                for x in range(bath.shape[1]):
                    noise[y, x] = pnoise2(y/freq, x/freq, octaves=10)

            bath += 10*noise

        return bath


    def lsm(X, Y, **kwargs):
        # The land-sea mask for the idealised Beaufort Gyre
        # 0 : land
        # 1 : sea

        lsm = np.zeros(X.shape, dtype=np.float64)

        # Firstly centre the coordinates around the gyre centre
        Xp = X - param['GCx']
        Yp = Y - param['GCy']

        # Add the circular gyre
        # lsm[((Y-param['GCy'])**2 + (X-param['GCx']**2)) < param['GR']**2] = 1
        lsm[(Yp**2 + Xp**2) < param['GR']**2] = 1

        # Add the sponge region/channel
        # lsm[((X-param['GCx'])**2 < param['CRx']**2)*(Y < param['GCy'])] = 1
        lsm[(Xp**2 < param['CRx']**2)*(Yp < 0)] = 1

        # Ensure the lsm is closed
        lsm[0, :]  = 0
        lsm[-1, :] = 0
        lsm[:, 0]  = 0
        lsm[:, -1] = 0

        # Plot the land-sea mask (but with an option to avoid recursion)
        if {'plot'} <= kwargs.keys():
            if kwargs['plot']:
                plot_field(X, Y, lsm, cm.gray, 'lsm.png', mask=False)

        return lsm

    def bath(X, Y):
        # Bathymetry for the idealised beaufort_gyre (linear slopes)

        bath = gen_bath(X,Y)

        # Plot the bathymetry
        plot_field(X, Y, bath, cm.deep, 'bath.png', mask=True)

        return bath

    def init_h1(X, Y):
        # Initial layer 1 thickness

        h1_init = param['H1']*np.ones(X.shape, dtype=np.float64)

        # Plot the initial layer 1 thickness region
        plot_field(X, Y, h1_init, cm.balance, 'init_h1.png', mask=True)

        return h1_init

    def init_h2(X, Y):
        # Initial layer 2 thickness

        h2_init = param['H2']*np.ones(X.shape, dtype=np.float64)

        # Plot the sponge region
        plot_field(X, Y, h2_init, cm.balance, 'init_h2.png', mask=True)

        return h2_init

    def init_h3(X, Y):
        # Initial layer 3 thickness (calculated by subtracting layer 1+2 thickness
        # from the bathymetry)

        h1_init = param['H1']*np.ones(X.shape, dtype=np.float64)
        h2_init = param['H2']*np.ones(X.shape, dtype=np.float64)
        bath    = gen_bath(X, Y)

        h3_init = bath - h1_init - h2_init

        # Plot the sponge region
        plot_field(X, Y, h3_init, cm.balance, 'init_h3.png', mask=True)

        return h3_init


    def tau_x_poly(X, Y):
        # Wind stress (x component) - polynomial formulation
        # Firstly construct the wind stress within the gyre domain

        # Transform to polar coordinates around the gyre centre
        Xp = X - param['GCx']
        Yp = Y - param['GCy']
        r     = np.sqrt(Xp**2 + Yp**2)
        theta = np.arctan2(Yp, Xp)

        # Define parameters
        b = 1/param['GR']
        d = np.sqrt(2/3)
        a = (4*param['tau_max'])/(d*param['GR']*(2-(b*d*param['GR'])**2))

        # Calculate wind stress
        tau_phi = (a*r/4)*(2-((b**2)*(r**2)))

        tau_x   = tau_phi*np.sin(theta)

        # Now construct the wind stress outside of the gyre
        c = (a*(param['GR']**2)/4)*(2-((b**2)*(param['GR']**2)))

        tau_x_supp = c*(Yp/r**2)
        tau_x[Yp < -param['GR']] = tau_x_supp[Yp < -param['GR']]

        # Plot the x component of the wind stress
        plot_field(X, Y, tau_x, cm.balance, 'tau_x.png',
                   vmin=-0.1, vmax=0.1, mask=True)

        return tau_x

    def tau_y_poly(X, Y):
        # Wind stress (y component) - polynomial formulation
        # Firstly construct the wind stress within the gyre domain

        # Transform to polar coordinates around the gyre centre
        Xp = X - param['GCx']
        Yp = Y - param['GCy']
        r     = np.sqrt(Xp**2 + Yp**2)
        theta = np.arctan2(Yp, Xp)

        # Define parameters
        b = 1/param['GR']
        d = np.sqrt(2/3)
        a = (4*param['tau_max'])/(d*param['GR']*(2-(b*d*param['GR'])**2))

        # Calculate wind stress
        tau_phi = (a*r/4)*(2-((b**2)*(r**2)))

        tau_y   = -tau_phi*np.cos(theta)

        # Now construct the wind stress outside of the gyre
        c = (a*(param['GR']**2)/4)*(2-((b**2)*(param['GR']**2)))

        tau_y_supp = -c*(Xp/r**2)
        tau_y[Yp < -param['GR']] = tau_y_supp[Yp < -param['GR']]

        # Plot the x component of the wind stress
        plot_field(X, Y, tau_y, cm.balance, 'tau_y.png',
                   vmin=-0.1, vmax=0.1, mask=True)

        return tau_y

    def tau_curl():
        # Also plot the vertical component of the wind stress curl (for
        # diagnostic purposes, not used in Aronnax code)

        X = Grid(nx=param['nx'], ny=param['ny'], layers=param['layers'],
                 dx=param['res'], dy=param['res']).x
        Y = Grid(nx=param['nx'], ny=param['ny'], layers=param['layers'],
                 dx=param['res'], dy=param['res']).y

        X, Y = np.meshgrid(X, Y)

        taux = tau_x_poly(X, Y)
        tauy = tau_y_poly(X, Y)

        dtauy_dx = np.gradient(tauy, 5000, axis=1)
        dtaux_dy = np.gradient(taux, 5000, axis=0)

        tau_curl = (dtauy_dx - dtaux_dy)

        # Plot the wind stress curl
        # Plot against the empirical maximum (tau_curl_max = -a):
        # Define parameters
        b = 1/param['GR']
        d = np.sqrt(2/3)
        a = (4*param['tau_max'])/(d*param['GR']*(2-(b*d*param['GR'])**2))

        plot_field(X, Y, tau_curl, cm.curl, 'tau_curl.png',
                   vmin = -a, vmax = +a, mask=True)


    def sponge_h1(X, Y):
        # Relax the layer 1 thickness towards the default value within the sponge
        # region

        sponge_h1 = param['H1']*np.ones(X.shape, dtype=np.float64)

        # Plot the sponge region
        plot_field(X, Y, sponge_h1, cm.balance, 'sponge_h1.png', mask=True)

        return sponge_h1

    def sponge_h1_ts(X, Y):
        # Set the layer 1 sponge relaxation timescale within the sponge region

        # Firstly centre the coordinates around the gyre centre
        # Xp = X - param['GCx']
        # Yp = Y - param['GCy']

        C_g = param['GCy'] - param['GR'] # Northern limit of channel

        sponge_h1_ts = np.zeros(Y.shape, dtype=np.float64)
        sponge_h1_ts = 0.5*(1 - np.tanh((2*np.pi/param['grad_Ly'])*(Y - C_g + param['grad_Ly']/3)))
        sponge_h1_ts[Y > C_g] = 0
        sponge_h1_ts /= param['sponge_ts']*24*3600

        # Plot the sponge timescale
        plot_field(X, Y, sponge_h1_ts, cm.balance, 'sponge_h1_ts.png', mask=True)

        return sponge_h1_ts

    def sponge_h2(X, Y):
        # Relax the layer 2 thickness towards the default value within the sponge
        # region

        sponge_h2 = param['H2']*np.ones(X.shape, dtype=np.float64)

        # Plot the sponge region
        plot_field(X, Y, sponge_h2, cm.balance, 'sponge_h2.png', mask=True)

        return sponge_h2

    def sponge_h2_ts(X, Y):
        # Set the layer 2 sponge relaxation timescale within the sponge region

        # Firstly centre the coordinates around the gyre centre
        Xp = X - param['GCx']
        Yp = Y - param['GCy']

        sponge_h2_ts = np.zeros(X.shape, dtype=np.float64)
        sponge_h2_ts[Yp < -param['GR']] = 1/(3600*24*param['sponge_ts'])

        # Plot the sponge timescale
        plot_field(X, Y, sponge_h2_ts, cm.balance, 'sponge_h2_ts.png', mask=True)

        return sponge_h2_ts

    def sponge_h3(X, Y):
        # Relax the layer 3 thickness towards the default value within the sponge
        # region (but this sponge is inactive)

        sponge_h3 = np.zeros(X.shape, dtype=np.float64)

        # Plot the sponge region
        plot_field(X, Y, sponge_h3, cm.balance, 'sponge_h3.png', mask=True)

        return sponge_h3

    def sponge_h3_ts(X, Y):
        # Set the layer 3 sponge relaxation timescale within the sponge region
        # (but this sponge is inactive)

        sponge_h3_ts = np.zeros(X.shape, dtype=np.float64)

        # Plot the sponge timescale
        plot_field(X, Y, sponge_h3_ts, cm.balance, 'sponge_h3_ts.png', mask=True)

        return sponge_h3_ts


    def f0(X, Y):
        # Set the coriolis parameter
        f_param = np.sin(param['lat']*np.pi/180)*4*np.pi/(86400)
        f0 = f_param*np.ones(X.shape, dtype=np.float64)

        # Plot the coriolis parameter
        plot_field(X, Y, f0, cm.balance, 'f0.png', mask=True)

        return f0

    def wind_ramp_up(nTimeSteps, dt):
        # Function to gradually ramp up the wind forcing from stationary
        # t_crit = time at which wind is at full strength
        t_crit = 86400*100
        alpha = np.arctanh(0.999999)/t_crit

        t = dt*np.arange(nTimeSteps)
        wstr = 0.5*np.tanh(alpha*(t-(t_crit/2)))+0.5

        f, ax = plt.subplots(1, 1, figsize=(5,10))
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Relative wind strength')
        ax.plot(t/(3600*24*360), wstr)
        plt.savefig(dirs['sim'] + '/figures/wind_ramp_up.png')
        plt.close()

        return wstr


    ###########################################################################
    # RUN ARONNAX                                                             #
    ###########################################################################

    with working_directory(dirs['sim']):
        tau_curl()
        drv.simulate(initHfile            = [init_h1, init_h2, init_h3],
                     zonalWindFile        = [tau_x_poly],
                     meridionalWindFile   = [tau_y_poly],
                     wind_mag_time_series_file = [wind_ramp_up],
                     wetMaskFile          = [lsm],
                     spongeHTimeScaleFile = [sponge_h1_ts, sponge_h2_ts, sponge_h3_ts],
                     spongeHFile          = [sponge_h1, sponge_h2, sponge_h3],
                     depthFile            = [bath],
                     fUfile               = [f0],
                     fVfile               = [f0],
                     layers               = param['layers'],
                     nx                   = param['nx'],
                     ny                   = param['ny'],
                     dx                   = param['res'],
                     dy                   = param['res'],
                     exe                  = 'aronnax_external_solver',
                     dt                   = param['dt'],
                     # Add a +1 to nTimeSteps so Aronnax saves the last frame
                     nTimeSteps           = int(np.ceil(86400.*param['sim_time']/
                                                    param['dt']))+1,
                     dumpFreq             = 86400.*param['snap_freq'],
                     avFreq               = 86400.*param['av_freq'],
                     checkpointFreq       = 86400.*param['chk_freq'],
                     )


if __name__ == '__main__':
    run_beaufort_linear3()
    convert_to_netcdf(dirs['sim'] + '/output/',
                      dirs['sim'] + '/netcdf-output/',
                      snap=True,
                      av=True)
