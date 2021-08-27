#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script visualises CTD/Mooring data from the Beaufort Gyre
@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cm
import cartopy.crs as ccrs
import pickle
import seawater as sw
from netCDF4 import Dataset
from skimage.measure import block_reduce
from scipy.interpolate import interp2d
from glob import glob

##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

dirs = {'root'  : os.path.dirname(os.path.realpath(__file__)) + '/'}
dirs['obs']   = dirs['root'] + 'obs/'

param = {'salt_lim' : 33.0}

##############################################################################
# PLOT LOCATIONS                                                             #
##############################################################################

if os.path.isfile(dirs['obs'] + 'obs_data.pkl'):
    with open(dirs['obs'] + 'obs_data.pkl', 'rb') as fh:
        data = pickle.load(fh)

        # Load salinity data
        with Dataset(dirs['obs'] + 'woa18_A5B7_s00_04.nc', mode='r') as nc:
            s150_data = nc.variables['s_an'][0, 22, :, :]
            s150_lon  = nc.variables['lon'][:]
            s150_lat  = nc.variables['lat'][:]

else:
    dir_list = sorted(glob(dirs['obs'] + '/*'))
    data = {'fh'   : [],
            'lon'  : [],
            'lat'  : [],
            'temp_pot' : [],
            'temp'     : [],
            'salt'     : [],
            'pressure' : [],
            'time'     : [],
            'in_gyre'  : [],
            'in_gyre2' : [],
            'density'  : [],
            'density_gradient' : []
            }

    i = 0

    # Discriminate between interior and exterior points
    # Load salinity data
    with Dataset(dirs['obs'] + 'woa18_A5B7_s00_04.nc', mode='r') as nc:
        s150_data = nc.variables['s_an'][0, 22, :, :]
        s150_lon  = nc.variables['lon'][:]
        s150_lat  = nc.variables['lat'][:]

    s150 = interp2d(s150_lon, s150_lat, s150_data)

    for directory in dir_list:
        fh_list = sorted(glob(directory + '/*'))

        for fh in fh_list:
            with Dataset(fh, mode='r') as nc:
                data['fh'].append(fh)

                try:
                    data['lon'].append(float(nc.variables['longitude'][0]))
                except:
                    data['lon'].append(np.nan)

                try:
                    data['lat'].append(float(nc.variables['latitude'][0]))
                except:
                    data['lat'].append(np.nan)

                if s150(data['lon'][-1], data['lat'][-1]) < param['salt_lim']:
                    data['in_gyre'].append(1)
                else:
                    data['in_gyre'].append(0)

                try:
                    data['temp_pot'].append(np.array(nc.variables['potential_temperature'][:]))
                except:
                    data['temp_pot'].append(np.nan)

                try:
                    data['temp'].append(np.array(nc.variables['temperature'][:]))
                except:
                    data['temp'].append(np.nan)

                try:
                    data['salt'].append(np.array(nc.variables['salinity'][:]))
                except:
                    data['salt'].append(np.nan)

                try:
                    data['pressure'].append(np.array(nc.variables['pressure'][:]))
                except:
                    data['pressure'].append(np.nan)

                try:
                    data['time'].append(float(nc.variables['time'][0]))
                except:
                    data['time'].append(np.nan)

                # Calculate the density
                data['density'].append(sw.dens0(s=data['salt'][-1],
                                               t=data['temp'][-1])-1000)

                data['density_gradient'].append(np.gradient(data['density'][-1]))

                data['in_gyre2'].append(np.ones_like(data['pressure'][-1])*data['in_gyre'][-1])

                i += 1
                print(i)

    with open(dirs['obs'] + 'obs_data.pkl', 'wb') as fh:
        pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)

# Plot the data locations
data_crs = ccrs.PlateCarree()
proj_crs = ccrs.Orthographic(central_longitude=0.0,
                             central_latitude=90.0)

f1 = plt.figure(figsize=(9,9))
ax1 = plt.subplot(111, projection=proj_crs)
f1.subplots_adjust(hspace=0.0, wspace=0.2, top=0.925, left=0.1)

ax1.set_global()
ax1.coastlines()
# ax1.stock_img()
ax1.set_extent([-180, -120, 70, 85],
               crs = data_crs)

salt_150m = ax1.pcolormesh(s150_lon, s150_lat, s150_data,
                           cmap=cm.haline,
                           vmin=32.4,
                           vmax=34.4,
                           transform=data_crs)

# Contour the 33PSU isopycnal / 150m depth interface
salt_150m_cont = ax1.contour(s150_lon, s150_lat, s150_data,
                             levels=np.array([param['salt_lim'],]),
                             colors='r', linewidths=1, linestyles='-',
                             transform=data_crs)
ax1.clabel(salt_150m_cont, fmt='%1.1f')

# Load bathymetry data
with Dataset(dirs['obs'] + 'gebco_2021_n90.0_s65.0_w-180.0_e-90.0.nc', mode='r') as nc:
    bath_data = nc.variables['elevation'][:],
    bath_data = block_reduce(bath_data[0], (5, 5), func=np.mean)
    bath_lon  = block_reduce(nc.variables['lon'][:], (5,), func=np.mean)
    bath_lat  = block_reduce(nc.variables['lat'][:], (5,), func=np.mean)

bath = ax1.contour(bath_lon, bath_lat, bath_data,
                   levels=np.arange(-5000, 500, 500),
                   colors='k', linewidths=0.5, linestyles='-',
                   transform=data_crs)

data_loc = ax1.scatter(data['lon'], data['lat'],
                       c=data['in_gyre'], cmap=plt.get_cmap('bwr'),
                       vmin=0, vmax=1,
                       s=10, marker='.',  transform=data_crs)

# Set up the colorbar
axpos = ax1.get_position()
pos_x = axpos.x0+axpos.width + 0.02
pos_y = axpos.y0
cax_width = 0.02
cax_height = axpos.height

pos_cax = f1.add_axes([pos_x, pos_y, cax_width, cax_height])

cb = plt.colorbar(salt_150m, cax=pos_cax)
cb.set_label('Salinity (PSU)', size=12)

plt.savefig('obs_location.png', dpi=300)
plt.close()

# Plot the density profiles
f2 = plt.figure(figsize=(18,9))
ax2 = plt.subplot(121)
ax3 = plt.subplot(122)
f1.subplots_adjust(hspace=0.0, wspace=0.05, top=0.925, left=0.1)

p_bins = np.linspace(0, 400, num=101)
drdp_bins = np.logspace(-2.5, -0.5, num=101)

p_list = np.concatenate(data['pressure']).ravel()
drdp_list = np.concatenate(data['density_gradient']).ravel()
in_gyre_list = np.concatenate(data['in_gyre2']).ravel()

p_list_gyre = p_list[in_gyre_list == 1]
p_list_notgyre = p_list[in_gyre_list == 0]
drdp_list_gyre = drdp_list[in_gyre_list == 1]
drdp_list_notgyre = drdp_list[in_gyre_list == 0]

drdp_hist_gyre = np.histogram2d(p_list_gyre, drdp_list_gyre,
                                bins=[p_bins, drdp_bins])[0]

drdp_hist_notgyre = np.histogram2d(p_list_notgyre, drdp_list_notgyre,
                                   bins=[p_bins, drdp_bins])[0]

for lev in range(np.shape(drdp_hist_gyre)[0]):
    drdp_hist_gyre[lev, :] /= np.max(drdp_hist_gyre[lev, :])
    drdp_hist_notgyre[lev, :] /= np.max(drdp_hist_notgyre[lev, :])

for i in range(len(data['temp'])):
    if data['pressure'][i][-1] > 100:
        if data['pressure'][i][0] < 40:
            # if data['in_gyre'][i]:
            #     color = 'r'
            # else:
            #     color = 'b'

            # ax2.plot(data['density'][i], data['pressure'][i],
            #           color=color, alpha=0.01)
            # ax3.plot(np.gradient(data['density'][i]),
            #          data['pressure'][i], color=color, alpha=0.01)
            ax2.pcolormesh(drdp_bins, p_bins, drdp_hist_gyre)
            ax3.pcolormesh(drdp_bins, p_bins, drdp_hist_notgyre)

# ax2.set_xlim([15, 30])
# ax2.set_ylim([-10, 400])
ax2.set_xscale('log')
ax2.invert_yaxis()
ax2.set_ylabel('Pressure (dbar)')
ax2.set_xlabel('Potential density gradient')
ax2.title.set_text('Within gyre')

ax3.set_xscale('log')
# ax3.set_xlim([0.002, 0.5])
# ax3.set_ylim([-10, 400])
ax3.invert_yaxis()
ax2.set_ylabel('Pressure (dbar)')
ax3.set_xlabel('Potential density gradient')
ax3.title.set_text('Outside of gyre')

plt.savefig('gradient.png', dpi=300)



print()
