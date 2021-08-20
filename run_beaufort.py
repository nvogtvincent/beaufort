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
# DIRECTORIES & PARAMETERS                                                   #
##############################################################################

dirs = {'root'  : os.path.dirname(os.path.realpath(__file__)) + '/'}
dirs['ctrl']   = dirs['root'] + 'ctrl/'
dirs['linear'] = dirs['root'] + 'linear/'
dirs['curved'] = dirs['root'] + 'curved/'

param = {'res'  :    5e3,     # Resolution in metres
         'Lx'   : 1510e3,     # Domain width in metres
         'Ly'   : 1710e3,     # Domain height in metres
         'R'    :  750e3,     # Gyre radius in metres
         'Cx'   :  755e3,     # Gyre centre (x coordinate)
         'Cy'   :  955e3,     # Gyre centre (y coordinate)
         'Wx'   :  100e3,     # Channel radius in metres
         }

def bg_lsm(X, Y):
    """
    The land-sea mask for the idealised Beaufort Gyre
    0 : land
    1 : sea
    """

    lsm = np.zeros(X.shape, dtype=np.float64)

    # Add the circular gyre
    lsm[((Y-param['Cy'])**2 + (X-param['Cx']**2)) < param['R']**2] = 1

    # Add the sponge region/channel
    lsm[((X-param['Cx'])**2 < param['Wx']**2)*(Y < param['Cy'])] = 1

    # Ensure the lsm is closed
    lsm[0, :]  = 0
    lsm[-1, :] = 0
    lsm[:, 0]  = 0
    lsm[:, -1] = 0

    # Plot the land-sea mask
    f, ax = plt.subplots(1, 1, figsize=(5, 10))
    ax.pcolormesh(X/1e3, Y/1e3, lsm, cmap=cm.gray)
    plt.xlim(0,1500)
    plt.axes().set_aspect('equal')
    plt.xlabel('x coordinate (km)')
    plt.ylabel('y coordinate (km)')
    plt.savefig(dirs['ctrl'] + 'figures/lsm.png')
    plt.close()

    return lsm


# import os.path as p

# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# self_path = p.dirname(p.abspath(__file__))
# root_path = p.dirname(self_path)

# import sys
# sys.path.append(p.join(root_path, 'test'))
# sys.path.append(p.join(root_path, 'reproductions/Davis_et_al_2014'))

# import aronnax.driver as drv
# from aronnax.utils import working_directory



