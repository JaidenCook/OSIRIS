#!/usr/bin/python

__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "0.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"

#%matplotlib notebook
import os,sys
import time
from datetime import datetime
import glob
import shutil
import re
from math import pi
import warnings
import subprocess
warnings.filterwarnings("ignore")

# Array stuff:
import numpy as np
warnings.simplefilter('ignore', np.RankWarning)

# Plotting stuff:
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 12})

plt.rc('xtick', color='k', labelsize='medium', direction='out')
plt.rc('xtick.major', size=6, pad=4)
plt.rc('xtick.minor', size=4, pad=4)

plt.rc('ytick', color='k', labelsize='medium', direction='out')
plt.rc('ytick.major', size=6, pad=4)
plt.rc('ytick.minor', size=4, pad=4)

# Parser options:
from optparse import OptionParser

# Scipy stuff:
import scipy
from scipy.fft import fftn,fftfreq,fftshift,ifftshift,ifftn
from scipy import stats
import scipy.optimize as opt

# Astropy stuff:
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy import wcs
from astropy.io import fits
from astropy.io.votable import parse_single_table
from astropy.table import Table,Column,vstack
from astropy.io.votable import writeto as writetoVO
from astropy.visualization import astropy_mpl_style
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from scipy import ndimage, misc

# MWA beam stuff
from mwa_pb import primary_beam as pb

import Osiris


def calc_21cm_pspec(freq_cent,channel,u_lam_arr,v_lam_arr,eta,test_plot=False,cosmo=None):

    from scipy import interpolate

    if cosmo == None:
        # Default case.
        # Importing the cosmology. Using the latest Planck 2018 results.
        from astropy.cosmology import Planck18

        cosmo = Planck18
    else:
        # User inputted cosmology, should still be an astropy.cosmology object.
        pass    

    c = 299792458.0/1000 #[km/s]
    parent_path = '/home/jaiden/Documents/EoR/21cmpspec/model/'
    file_seed = parent_path + 'ps_no_halos_z'

    N_char = len(file_seed)

    # Channel bandwidth:
    dnu = channel.bandwidth # MHz
    dnu_f = channel.fine_chan_width/1e+6 #MHz
    f21 = (1000*c)/(0.21) #[Hz]
    z = (f21)/freq_cent - 1

    filenames = os.popen('find {0}*'.format(file_seed)).read().split('\n')[:-1]

    redshift_vec = np.array([float(file[N_char:(N_char + 6)]) for file in filenames])
    pspec_names = [file[len(parent_path):] for file in filenames]

    file_index = np.argmin(np.abs(redshift_vec-z))
    file_cent = pspec_names[file_index]

    # Calculating the cosmological conversion factor.
    C_factor = Osiris.Power_spec.Power2Tb(dnu*1e+6,dnu_f*1e+6,freq_cent,z,cosmo=cosmo)

    # Allocating the appropriate simualted PS from the model folder.
    Data_array = np.loadtxt(parent_path + file_cent, usecols=(0,1,2))

    # Initialsing the power, and kr arrays.
    k_r = Data_array.T[0,:]
    #Norm_factor = (k_r**3 / (2*np.pi**2))*(1/C_factor)
    Norm_factor = ((2*np.pi**2)/k_r**3)*(1/C_factor)
    Power1D = Data_array.T[1,:]*Norm_factor
    u_PS = Data_array.T[2,:]

    # Keeping for comparison purposes.
    #Power1D_original = Data_array.T[1,:]*(k_r**3 / (2*np.pi**2))
    Power1D_original = Data_array.T[1,:]*((2*np.pi**2)/k_r**3)

    # Interpolating the power spectrum with respect to kr
    f = interpolate.interp1d(k_r,Data_array.T[1,:], kind='cubic')
    
    # Cosmological scaling parameter:
    h = cosmo.H(0).value/100 # Hubble parameter.
    E_z = cosmo.efunc(z) ## Scaling function, see (Hogg 2000)

    # Cosmological distances:
    Dm = cosmo.comoving_distance(z).value*h #[Mpc/h] Transverse co-moving distance.
    DH = 3000 # [Mpc/h] Hubble distance.

    # Getting kx, ky, and kz:
    k_x_grid = 2*np.pi*u_lam_arr/Dm
    k_y_grid = 2*np.pi*v_lam_arr/Dm
    k_z = eta * (2*np.pi*E_z*f21)/(DH*(1 + z)**2) # [Mpc^-1 h]

    # Creating the k_radius grid.
    k_r_grid = np.array([np.sqrt(k_x_grid**2 + k_y_grid**2 + kz**2) for kz in k_z]).T

    # Some modes are less, we will have no power in these modes.
    grid_ind = np.logical_and(k_r_grid >= np.min(k_r),k_r_grid < np.max(k_r))

    # Calculating the conversion factor.
    #Conversion_factor = ((k_r_grid[grid_ind])**3 / (2*np.pi**2))*(1/C_factor)#*(1e+6)
    Conversion_factor = ((2*np.pi**2)/(k_r_grid[grid_ind])**3)*(1/C_factor)#*(1e+6)

    sigma_arr = np.zeros(k_r_grid.shape)
    sigma_arr[grid_ind] = np.sqrt(Conversion_factor*f(k_r_grid[grid_ind]))

    temp_ind = np.logical_or(np.isnan(sigma_arr),grid_ind==False)
    sigma_arr[temp_ind] = 0.0

    # Specifying the weights array for calculating the power spectrum.
    weights_arr = np.ones(k_r_grid.shape)
    weights_arr[temp_ind] = 0.0

    # Initialising the 21cm sky visibility cube.
    Vis_21cm_cube = np.zeros(k_r_grid.shape,dtype=np.complex)

    np.real(Vis_21cm_cube)[temp_ind==False] = np.random.normal(0.0, sigma_arr[temp_ind==False],sigma_arr[temp_ind==False].size)/np.sqrt(2)
    np.imag(Vis_21cm_cube)[temp_ind==False] = np.random.normal(0.0, sigma_arr[temp_ind==False],sigma_arr[temp_ind==False].size)/np.sqrt(2)

    del sigma_arr

    #test_plot = True

    if test_plot:
        print('Plotting test 21cm plot.')
        # If True we want to test to see if we recover the 21cm power spectrum from the input model.
        dnu_fine = channel.bandwidth/len(channel.fine)

        Power = Osiris.Power_spec(Vis_21cm_cube,eta,u_lam_arr,v_lam_arr,freq_cent,dnu,dnu_fine)

        Power.Spherical()

        fig, axs = plt.subplots(1, figsize = (8,6))

        Osiris.Power_spec.plot_spherical(k_r,Power1D_original,figaxs=(fig,axs),lw=3,label='21CMFAST',step=False, color='k')
        Osiris.Power_spec.plot_spherical(Power.k_r,Power.Power1D,figaxs=(fig,axs),ls='--',lw=3,label='Reconstruction',step=False, color='r')

        plt.legend(fontsize=18)
        plt.tight_layout()

        plt.xlim([1e-2,4])

        plt.savefig('21cm-test-pspec.png')
    else:
        pass

    return Vis_21cm_cube




