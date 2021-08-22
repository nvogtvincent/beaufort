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

##############################################################################
# WIND STRESS FORMULATION                                                    #
##############################################################################
'''
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

As in Davis et al. (2014), outside of the gyre, we set the wind stress curl to
the following zero-curl field:

    tau_x = c(y/r^2)
    tau_y = c(-x/r^2)

Where c = (1/4)*(aR^2) + (1/(32pi^2))*(aR) for continuity with the gyre winds

'''
##############################################################################
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

dirs = {'root'  : os.path.dirname(os.path.realpath(__file__)) + '/'}
dirs['ctrl']   = dirs['root'] + 'ctrl'
dirs['linear'] = dirs['root'] + 'linear'
dirs['curved'] = dirs['root'] + 'curved'

param = {'res'       :    5e3,     # Resolution in metres
         'Lx'        : 1510e3,     # Domain width in metres
         'Ly'        : 1710e3,     # Domain height in metres
         'GR'        :  750e3,     # Gyre radius in metres
         'GCx'       :  755e3,     # Gyre centre (x coordinate)
         'GCy'       :  955e3,     # Gyre centre (y coordinate)
         'CRx'       :  100e3,     # Channel radius in metres

         'Lz'        :  2.0e3,     # Domain depth in metres
         'H1'        :  0.4e3,     # Layer 1 depth in metres

         'tau_max'   :  0.2e0,     # Maximum wind stress (N/m^2)

         'sponge_ts' :    1e0,     # Sponge timescale (years)

         'lat'       :    90.,     # Latitude (deg) to calculate f at

         'dt'        :    50.,     # Time step (seconds)
         'nt'        :    100,     # Number of time steps
         'out_freq'  :     1.,     # Output frequency (days)
         }


def run_beaufort_ctrl():

    ###########################################################################
    # SET UP INPUTS                                                           #
    ###########################################################################

    def lsm(X, Y):
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

        # Plot the land-sea mask
        f, ax = plt.subplots(1, 1, figsize=(5, 10))
        ax.pcolormesh(X/1e3, Y/1e3, lsm, cmap=cm.gray)
        # plt.xlim(0,1500)
        plt.axes().set_aspect('equal')
        plt.xlabel('x coordinate (km)')
        plt.ylabel('y coordinate (km)')
        plt.savefig(dirs['ctrl'] + '/figures/lsm.png')
        plt.close()

        return lsm

    def bath_ctrl(X, Y):
        # Bathymetry for the idealised beaufort_gyre (control, flat)

        bath = param['Lz']*np.ones(X.shape, dtype=np.float64)

        # Plot the bathymetry
        f, ax = plt.subplots(1, 1, figsize=(5, 10))
        ax.pcolormesh(X/1e3, Y/1e3, bath, cmap=cm.balance)
        # plt.xlim(0,1500)
        plt.axes().set_aspect('equal')
        plt.xlabel('x coordinate (km)')
        plt.ylabel('y coordinate (km)')
        plt.savefig(dirs['ctrl'] + '/figures/bath.png')
        plt.close()

        return bath

    def init_h1(X, Y):
        # Initial layer 1 thickness

        h1_init = param['H1']*np.ones(X.shape, dtype=np.float64)

        # Plot the initial layer 1 thickness region
        f, ax = plt.subplots(1, 1, figsize=(5, 10))
        ax.pcolormesh(X/1e3, Y/1e3, h1_init, cmap=cm.balance)
        # plt.xlim(0,1500)
        plt.axes().set_aspect('equal')
        plt.xlabel('x coordinate (km)')
        plt.ylabel('y coordinate (km)')
        plt.savefig(dirs['ctrl'] + '/figures/init_h1.png')
        plt.close()

        return h1_init

    def init_h2(X, Y):
        # Initial layer 2 thickness (calculated by subtracting layer 1 thickness
        # from the bathymetry)

        h1_init = param['H1']*np.ones(X.shape, dtype=np.float64)
        bath    = param['Lz']*np.ones(X.shape, dtype=np.float64)

        h2_init = h1_init - bath

        # Plot the sponge region
        f, ax = plt.subplots(1, 1, figsize=(5, 10))
        ax.pcolormesh(X/1e3, Y/1e3, h2_init, cmap=cm.balance)
        # plt.xlim(0,1500)
        plt.axes().set_aspect('equal')
        plt.xlabel('x coordinate (km)')
        plt.ylabel('y coordinate (km)')
        plt.savefig(dirs['ctrl'] + '/figures/init_h2.png')
        plt.close()

        return h2_init


    def tau_x(X, Y):
        # Wind stress (x component)
        # Firstly construct the wind stress within the gyre domain

        # Transform to polar coordinates around the gyre centre
        Xp = X - param['GCx']
        Yp = Y - param['GCy']
        r     = np.sqrt(Xp**2 + Yp**2)
        theta = np.arctan2(Yp, Xp)

        # Define parameters
        b = 2*np.pi/param['GR']

        # Calculate wind stress
        tau_phi = ((r/4) +
                   (1/(4*b))*np.sin(2*b*r) +
                   (1/(8*(b**2)*r))*np.cos(2*b*r))

        norm    = param['tau_max']/np.max(tau_phi[r < param['GR']])

        tau_x   = norm*tau_phi*np.sin(theta)

        # Now construct the wind stress outside of the gyre
        c = ((1/4)*(norm*param['GR']**2) +
             (1/(32*np.pi**2))*(norm*param['GR']))

        tau_x_supp = c*(Yp/r**2)
        tau_x[Yp < -param['GR']] = tau_x_supp[Yp < -param['GR']]

        # Plot the x component of the wind stress
        f, ax = plt.subplots(1, 1, figsize=(5, 10))
        ax.pcolormesh(X/1e3, Y/1e3, tau_x, cmap=cm.balance)
        # plt.xlim(0,1500)
        plt.axes().set_aspect('equal')
        plt.xlabel('x coordinate (km)')
        plt.ylabel('y coordinate (km)')
        plt.savefig(dirs['ctrl'] + '/figures/tau_x.png')
        plt.close()

        return tau_x

    def tau_y(X, Y):
        # Wind stress (x component)
        # Firstly construct the wind stress within the gyre domain

        # Transform to polar coordinates around the gyre centre
        Xp = X - param['GCx']
        Yp = Y - param['GCy']
        r     = np.sqrt(Xp**2 + Yp**2)
        theta = np.arctan2(Yp, Xp)

        # Define parameters
        b = 2*np.pi/param['GR']

        # Calculate wind stress
        tau_phi = ((r/4) +
                   (1/(4*b))*np.sin(2*b*r) +
                   (1/(8*(b**2)*r))*np.cos(2*b*r))

        norm    = param['tau_max']/np.max(tau_phi[r < param['GR']])

        tau_y   = -norm*tau_phi*np.cos(theta)

        # Now construct the wind stress outside of the gyre
        c = ((1/4)*(norm*param['GR']**2) +
             (1/(32*np.pi**2))*(norm*param['GR']))

        tau_y_supp = c*(-Xp/r**2)
        tau_y[Yp < -param['GR']] = tau_y_supp[Yp < -param['GR']]

        # Plot the x component of the wind stress
        f, ax = plt.subplots(1, 1, figsize=(5, 10))
        ax.pcolormesh(X/1e3, Y/1e3, tau_y, cmap=cm.balance)
        # plt.xlim(0,1500)
        plt.axes().set_aspect('equal')
        plt.xlabel('x coordinate (km)')
        plt.ylabel('y coordinate (km)')
        plt.savefig(dirs['ctrl'] + '/figures/tau_y.png')
        plt.close()

        return tau_y

    def sponge_h1(X, Y):
        # Relax the layer 1 thickness towards the default value within the sponge
        # region

        sponge_h = param['H1']*np.ones(X.shape, dtype=np.float64)

        # Plot the sponge region
        f, ax = plt.subplots(1, 1, figsize=(5, 10))
        ax.pcolormesh(X/1e3, Y/1e3, sponge_h, cmap=cm.balance)
        # plt.xlim(0,1500)
        plt.axes().set_aspect('equal')
        plt.xlabel('x coordinate (km)')
        plt.ylabel('y coordinate (km)')
        plt.savefig(dirs['ctrl'] + '/figures/sponge_h1.png')
        plt.close()

        return sponge_h

    def sponge_h1_ts(X, Y):
        # Set the layer 1 sponge relaxation timescale within the sponge region

        # Firstly centre the coordinates around the gyre centre
        Xp = X - param['GCx']
        Yp = Y - param['GCy']

        sponge_h_ts = np.zeros(X.shape, dtype=np.float64)
        sponge_h_ts[Yp < -param['GR']] = 1/(3600*24*360*param['sponge_ts'])

        # Plot the sponge timescale
        f, ax = plt.subplots(1, 1, figsize=(5, 10))
        ax.pcolormesh(X/1e3, Y/1e3, sponge_h_ts, cmap=cm.balance)
        # plt.xlim(0,1500)
        plt.axes().set_aspect('equal')
        plt.xlabel('x coordinate (km)')
        plt.ylabel('y coordinate (km)')
        plt.savefig(dirs['ctrl'] + '/figures/sponge_h1_ts.png')
        plt.close()

        return sponge_h_ts

    def sponge_h2(X, Y):
        # Relax the layer 2 thickness towards the default value within the sponge
        # region (but this sponge is inactive)

        sponge_h = np.zeros(X.shape, dtype=np.float64)

        # Plot the sponge region
        f, ax = plt.subplots(1, 1, figsize=(5, 10))
        ax.pcolormesh(X/1e3, Y/1e3, sponge_h, cmap=cm.balance)
        # plt.xlim(0,1500)
        plt.axes().set_aspect('equal')
        plt.xlabel('x coordinate (km)')
        plt.ylabel('y coordinate (km)')
        plt.savefig(dirs['ctrl'] + '/figures/sponge_h2.png')
        plt.close()

        return sponge_h

    def sponge_h2_ts(X, Y):
        # Set the layer 1 sponge relaxation timescale within the sponge region
        # (but this sponge is inactive)

        sponge_h_ts = np.zeros(X.shape, dtype=np.float64)

        # Plot the sponge timescale
        f, ax = plt.subplots(1, 1, figsize=(5, 10))
        ax.pcolormesh(X/1e3, Y/1e3, sponge_h_ts, cmap=cm.balance)
        # plt.xlim(0,1500)
        plt.axes().set_aspect('equal')
        plt.xlabel('x coordinate (km)')
        plt.ylabel('y coordinate (km)')
        plt.savefig(dirs['ctrl'] + '/figures/sponge_h2_ts.png')
        plt.close()

        return sponge_h_ts


    def f0(X, Y):
        # Set the coriolis parameter
        f_param = np.sin(param['lat']*np.pi/180)*4*np.pi/(86400)
        f0 = f_param*np.ones(X.shape, dtype=np.float64)

        # Plot the coriolis parameter
        f, ax = plt.subplots(1, 1, figsize=(5, 10))
        ax.pcolormesh(X/1e3, Y/1e3, f0, cmap=cm.balance)
        # plt.xlim(0,1500)
        plt.axes().set_aspect('equal')
        plt.xlabel('x coordinate (km)')
        plt.ylabel('y coordinate (km)')
        plt.savefig(dirs['ctrl'] + '/figures/sponge_h.png')
        plt.close()

        return f0

    # Run Aronnax
    with working_directory(dirs['ctrl']):
        drv.simulate(initHfile            = [init_h1, init_h2],
                     zonalWindFile        = [tau_x],
                     meridionalWindFile   = [tau_y],
                     wetMaskFile          = [lsm],
                     spongeHTimeScaleFile = [sponge_h1_ts, sponge_h2_ts],
                     spongeHFile          = [sponge_h1, sponge_h2],
                     depthFile            = [bath_ctrl],
                     fUfile               = [f0],
                     fVfile               = [f0],
                     layers               = 2,
                     nx                   = int(param['Lx']/param['res']),
                     ny                   = int(param['Ly']/param['res']),
                     dx                   = param['res'],
                     dy                   = param['res'],
                     exe                  = 'aronnax_external_solver',
                     dt                   = param['dt'],
                     nTimeSteps           = param['nt'],
                     dumpFreq             = 86400.*param['out_freq'],
                     avFreq               = 86400.*param['out_freq'],
                     checkpointFreq       = 86400.*param['out_freq']
)


if __name__ == '__main__':
    run_beaufort_ctrl()