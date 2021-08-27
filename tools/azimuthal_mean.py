#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script calculates the azimuthal mean from aronnax output fields
(https://github.com/edoddridge/aronnax)
@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cm
from matplotlib.colors import LightSource
import matplotlib.animation as ani
from netCDF4 import Dataset

##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

dirs = {'root'  : os.path.dirname(os.path.realpath(__file__)) + '/../'}
dirs['ctrl']   = dirs['root'] + 'ctrl'
dirs['linear'] = dirs['root'] + 'linear'
dirs['curved'] = dirs['root'] + 'curved'

param = {# PLOTTING CHOICES
         'n_rad'    :                100,  # Number of radial sample points
         'n_azi'    :                360,  # Number of azimuthal sample points
         'y_min'    :                  1,  # First year to average

         'in_name'  :     'ctrl_test.nc',  # netcdf input name
         }

fh = {'in'  : dirs['ctrl'] + '/netcdf-output/' + param['in_name']}
fh['out'] = fh['in'].rstrip('.nc') + '_azimuthal_mean.nc'

##############################################################################
# IMPORT DATA                                                                #
##############################################################################

with Dataset(fh['in'], mode='r') as nc:
    x = nc.variables['x'][:]
    xp1 = nc.variables['xp1'][:]
    y = nc.variables['y'][:]
    yp1 = nc.variables['xp1'][:]

    h = nc.variables['h_snap'][:, param['layer'], :, :]
    sq1 = np.shape(h)[1] - np.shape(h)[2]
    sq2 = np.shape(h)[2]

    if param['c_var'] == 'vort':
        u = nc.variables['u_snap'][:, param['layer'], sq1:, :sq2]
        v = nc.variables['v_snap'][:, param['layer'], sq1+1:, :sq2]

    time = nc.variables['time_snap'][:]

