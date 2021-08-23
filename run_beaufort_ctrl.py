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

dirs = {'root'  : os.path.dirname(os.path.realpath(__file__)) + '/'}
dirs['ctrl']   = dirs['root'] + 'ctrl'
dirs['linear'] = dirs['root'] + 'linear'
dirs['curved'] = dirs['root'] + 'curved'

param = {# DOMAIN GEOMETRY
         'res'       :    5e3,     # Resolution in metres
         'Lx'        : 1510e3,     # Domain width in metres
         'Ly'        : 1710e3,     # Domain height in metres
         'GR'        :  750e3,     # Gyre radius in metres
         'GCx'       :  755e3,     # Gyre centre (x coordinate)
         'GCy'       :  955e3,     # Gyre centre (y coordinate)
         'CRx'       :  100e3,     # Channel radius in metres

         'Lz'        :  2.0e3,     # Domain depth in metres
         'H1'        :  0.4e3,     # Layer 1 depth in metres

         # WIND FORCING
         'tau_max'   :   1e-1,     # Maximum wind stress (N/m^2)

         # SPONGE REGION
         'sponge_ts' :    3e1,     # Sponge timescale (days)

         # CORIOLIS
         'lat'       :    90.,     # Latitude (deg) to calculate f at

         # NUMERICS
         'dt'        :    50.,     # Time step (seconds)
         'nt'        :51840*12,     # Number of time steps
         'out_freq'  :     1.,     # Output frequency (days)
         }


def run_beaufort_ctrl():

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
        plt.savefig(dirs['ctrl'] + '/figures/' + fn)
        plt.close()

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

    def bath_ctrl(X, Y):
        # Bathymetry for the idealised beaufort_gyre (control, flat)

        bath = param['Lz']*np.ones(X.shape, dtype=np.float64)

        # Plot the bathymetry
        plot_field(X, Y, bath, cm.balance, 'bath.png', mask=True)

        return bath

    def init_h1(X, Y):
        # Initial layer 1 thickness

        h1_init = param['H1']*np.ones(X.shape, dtype=np.float64)

        # Plot the initial layer 1 thickness region
        plot_field(X, Y, h1_init, cm.balance, 'init_h1.png', mask=True)

        return h1_init

    def init_h2(X, Y):
        # Initial layer 2 thickness (calculated by subtracting layer 1 thickness
        # from the bathymetry)

        h1_init = param['H1']*np.ones(X.shape, dtype=np.float64)
        bath    = param['Lz']*np.ones(X.shape, dtype=np.float64)

        h2_init = bath - h1_init

        # Plot the sponge region
        plot_field(X, Y, h2_init, cm.balance, 'init_h2.png', mask=True)

        return h2_init


    def tau_x_trig(X, Y):
        # Wind stress (x component) - trig formulation
        # Firstly construct the wind stress within the gyre domain

        # Transform to polar coordinates around the gyre centre
        Xp = X - param['GCx']
        Yp = Y - param['GCy']
        r     = np.sqrt(Xp**2 + Yp**2)
        theta = np.arctan2(Yp, Xp)

        # Define parameters
        b = np.pi/(2*param['GR'])

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
        plot_field(X, Y, tau_x, cm.balance, 'tau_x.png',
                   vmin=-0.1, vmax=0.1)

        return tau_x

    def tau_y_trig(X, Y):
        # Wind stress (y component) - trig formulation
        # Firstly construct the wind stress within the gyre domain

        # Transform to polar coordinates around the gyre centre
        Xp = X - param['GCx']
        Yp = Y - param['GCy']
        r     = np.sqrt(Xp**2 + Yp**2)
        theta = np.arctan2(Yp, Xp)

        # Define parameters
        b = np.pi/(2*param['GR'])

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
        plot_field(X, Y, tau_y, cm.balance, 'tau_y.png',
                   vmin=-0.1, vmax=0.1)

        return tau_y

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

        X = np.arange(param['res']/2, param['Lx']+param['res']/2, param['res'])
        Y = np.arange(param['res']/2, param['Ly']+param['res']/2, param['res'])
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
        Xp = X - param['GCx']
        Yp = Y - param['GCy']

        sponge_h1_ts = np.zeros(X.shape, dtype=np.float64)
        sponge_h1_ts[Yp < -param['GR']] = 1/(3600*24*360*param['sponge_ts'])

        # Plot the sponge timescale
        plot_field(X, Y, sponge_h1_ts, cm.balance, 'sponge_h1_ts.png', mask=True)

        return sponge_h1_ts

    def sponge_h2(X, Y):
        # Relax the layer 2 thickness towards the default value within the sponge
        # region (but this sponge is inactive)

        sponge_h2 = np.zeros(X.shape, dtype=np.float64)

        # Plot the sponge region
        plot_field(X, Y, sponge_h2, cm.balance, 'sponge_h2.png', mask=True)

        return sponge_h2

    def sponge_h2_ts(X, Y):
        # Set the layer 1 sponge relaxation timescale within the sponge region
        # (but this sponge is inactive)

        sponge_h2_ts = np.zeros(X.shape, dtype=np.float64)

        # Plot the sponge timescale
        plot_field(X, Y, sponge_h2_ts, cm.balance, 'sponge_h2_ts.png', mask=True)

        return sponge_h2_ts


    def f0(X, Y):
        # Set the coriolis parameter
        f_param = np.sin(param['lat']*np.pi/180)*4*np.pi/(86400)
        f0 = f_param*np.ones(X.shape, dtype=np.float64)

        # Plot the coriolis parameter
        plot_field(X, Y, f0, cm.balance, 'f0.png', mask=True)

        return f0

    ###########################################################################
    # RUN ARONNAX                                                             #
    ###########################################################################

    with working_directory(dirs['ctrl']):
        tau_curl()
        drv.simulate(initHfile            = [init_h1, init_h2],
                     zonalWindFile        = [tau_x_poly],
                     meridionalWindFile   = [tau_y_poly],
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
                     checkpointFreq       = 86400.*param['out_freq'],
                     )


if __name__ == '__main__':
    run_beaufort_ctrl()