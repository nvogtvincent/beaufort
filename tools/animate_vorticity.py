#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This animates the output of an aronnax simulation
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
         'layer'    :                      1, # Which layer to plot
         'c_var'    :                 'vort', # Which variable to colour

         'cmap'     :             cm.balance, # Colormap choice
         'vmin'     :                   200.,
         'vmax'     :                   600.,

         'in_name'  :         'ctrl_test.nc', # netcdf input name
         'out_name' :   'test_animation.mp4', # animation name

         # ANIMATION PROPERTIES
         'fps'      :                     20, # Frames per second

         # SIMULATION PROPERTIES
         }

fh = {'in'  : dirs['ctrl'] + '/netcdf-output/' + param['in_name'],
      'out' : dirs['ctrl'] + '/animations/' + param['out_name']}

# Check if animations directory exists
if not os.path.isdir(dirs['ctrl'] + '/animations'):
    os.makedirs(dirs['ctrl'] + '/animations')

##############################################################################
# IMPORT DATA AND CALCULATE VORTICITY                                        #
##############################################################################

with Dataset(fh['in'], mode='r') as nc:
    x = nc.variables['x'][:]
    y = nc.variables['y'][:]

    h = nc.variables['h_snap'][:, param['layer'], :, :]
    sq1 = np.shape(h)[1] - np.shape(h)[2]
    sq2 = np.shape(h)[2]

    if param['c_var'] == 'vort':
        u = nc.variables['u_snap'][:, param['layer'], sq1:, :sq2]
        v = nc.variables['v_snap'][:, param['layer'], sq1+1:, :sq2]

    time = nc.variables['time_snap'][:]

# Quick hacky method to calculate vorticity - ONLY for prototype viz purposes!

dx = x[1] - x[0]
dy = dx

dvdx = np.gradient(v, dx, axis=2)
dudy = np.gradient(u, dy, axis=1)

vort = dvdx[:, :, :] - dudy[:, :, :]
f0 = (4*np.pi/86400)
vort /= f0

# Calculate a mask for neatness
x1 = np.arange(len(x)) - (len(x)-1)/2
y1 = np.copy(x1)
mask_rad = np.floor(x1[-1])-1
x1, y1 = np.meshgrid(x1, y1)
circ_mask = np.ones_like(x1)
circ_mask[x1**2 + y1**2 < mask_rad**2] = 0
vort = np.ma.masked_array(vort, mask=np.tile(circ_mask, (len(time), 1, 1)))

# Make a neat border
xcirc = mask_rad*np.cos(np.linspace(0, 2*np.pi, num=360)) + (len(x)-1)/2
ycirc = mask_rad*np.sin(np.linspace(0, 2*np.pi, num=360)) + (len(x)-1)/2

##############################################################################
# Set up plot ################################################################
##############################################################################

f, ax = plt.subplots(1, 1, figsize=(10, 10))

csfont = {'fontname': 'Ubuntu',
          'color'   : 'white'}
f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax.set_facecolor('k')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

ls = LightSource(azdeg=315, altdeg=30)

vort_viz = ls.shade(vort[0, :, :], cmap=cm.delta, blend_mode='soft',
                    vert_exag=1000, vmin=param['vmin'], vmax=param['vmax'])

im = ax.imshow(vort_viz, aspect='auto')
date = ax.text(5, 10, 'Day 0', fontsize='20', **csfont)

ax.plot(xcirc, ycirc, 'k-', linewidth = 3)

##############################################################################
# Set up animation ###########################################################
##############################################################################

def animate_gyre(t):
    vort_viz = ls.shade(vort[t, :, :], cmap=cm.delta, blend_mode='soft',
                        vert_exag=1000, vmin=-0.1, vmax=0.1)
    im.set_array(vort_viz)

    # Calculate the current time text
    day = int(time[t]/86400)
    day_text = 'Day ' + str(day)

    date.set_text(day_text)

    # Progress indicator
    if t%10 == 0:
        print('Progress: ' + str(int(t*100/len(time))) + '%')

    return [im, date]

##############################################################################
# Run the animation ##########################################################
##############################################################################


animation = ani.FuncAnimation(f,
                              animate_gyre,
                              frames=len(time),
                              interval=1000/param['fps'])

animation.save(param['out_name'],
               fps=param['fps'],
               bitrate=16000,)
