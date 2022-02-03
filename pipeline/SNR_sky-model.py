#!/usr/bin/python

__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "0.0.2"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"

# Generic stuff:
#%matplotlib notebook
import os,sys
import time
from datetime import datetime
import logging
import sys
import glob
import shutil
import re
from math import pi
import warnings
import subprocess
from numpy.core.defchararray import index

from traitlets.traitlets import default
warnings.filterwarnings("ignore")

# Array stuff:
import numpy as np
#warnings.simplefilter('ignore', np.RankWarning)

# Plotting stuff:
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

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
from scipy import signal
from scipy.fft import fftn,ifftn,fftfreq,fftshift,ifftshift
from scipy import stats
import scipy.optimize as opt

# Astropy stuff:
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy import wcs
from astropy.io import fits
from astropy.io import ascii
from astropy.io.votable import parse_single_table
from astropy.table import Table,Column,vstack
from astropy.io.votable import writeto as writetoVO

# MWA beam stuff
from mwa_pb import primary_beam as pb

sys.path.append(os.path.abspath("/home/jaiden/Documents/EoR/OSIRIS"))
import Osiris

import SNR_MWAbeam
import SNR_powerspec

def fits_split(hdu,column):
    """
    Takes the input header from a fits or metafits file, and extracts the column. 
    This function splits the column into a list of integers. This is specifically
    used for the 'DELAYS' and 'CHANNELS' columns. outputs a 1D numpy array.
    """

    column_vec = np.array([int(col) for col in hdu[column].split(",")])

    return column_vec

if __name__ == '__main__':
    # Defining the parser:
    usage="Usage: %prog [options]\n"
    parser = OptionParser(usage=usage)
    parser.add_option('--obsid',dest="obsid",default=None,help="Input the observation obsid.\n")
    parser.add_option('--uvmax',dest="uvmax",default=300,help="(u,v) max, determines image size, Npix = 4*uvmax + N_ker_size.\n")
    parser.add_option('--kernel_size',dest="N_ker_size",default=31,help="Degridding kernel size, default is 31 pixels.\n")
    parser.add_option('--all_sky',dest="all_sky",default=False,action='store_true',help="Make an all-sky model.\n")
    parser.add_option('--just_cenA',dest="just_cenA",default=False,action='store_true',help="Make only a model of Centuarus A.\n")
    parser.add_option('--just_crab',dest="just_crab",default=False,action='store_true',help="Make only a model of the Crab nebula.\n")
    parser.add_option('--no_cenA',dest="no_cenA",default=False,action='store_true',help="Exclude CenA from the all-sky model.\n")
    parser.add_option('--no_pupA',dest="no_pupA",default=False,action='store_true',help="Exclude Pupis A from the all-sky model.\n")
    parser.add_option('--no_vela',dest="no_vela",default=False,action='store_true',help="Exclude Vela from the all-sky model.\n")
    parser.add_option('--no_crab',dest="no_crab",default=False,action='store_true',help="Exclude the Crab nebula from the all-sky model.\n")
    parser.add_option('--save_partial_models',dest="save_part_mods",default=False,action='store_true',help="Make the all-sky model in steps.\n")
    parser.add_option('--save_plots',dest="save_plots",default=False,action='store_true',help="Save plots of the sky-model.\n")


    (options, args) = parser.parse_args()

    parent_path = '/home/jaiden/Documents/EoR/SNR-Pipeline/models/'
    out_path = '/home/jaiden/Documents/EoR/SNR-Pipeline/output/'

    ## Constants
    # Natural Constants:
    c = 299792458.0 #[m/s]

    # Sky model array dimensions.
    L = 2
    M = 2

    # Max absolute uv:
    #uvmax = 300 # [lambda]
    uvmax = int(options.uvmax) # [lambda]

    # Number of elements in the grid.
    N_ker_size = int(options.N_ker_size)
    N = 4*uvmax + N_ker_size # Works for all sky images.
    

    if options.obsid != None:
        # Loading in FITS header information from the OBSID metafits file.
        obsid = int(options.obsid)
        hdu = fits.getheader('{0}{1}.metafits'.format(parent_path,obsid))
        chans = fits_split(hdu,'CHANNELS')[:12] # Coarse channels.
        # Initialising file name list:
        name = '%s_' % obsid
    else:
        obsid = 1080136736
        chans = np.array([131,132,133,134,135,136,137,138,139,140,141,142])
        name = '%s_' % obsid

    #
    ## Creating (l,m) and frequency grids:
    #

    # Coarse channels for the observation.
    freqs = chans*1.28e+6
    waves = c/freqs

    channel = SNR_MWAbeam.channels(freqs)
    channel.calc_fine_chans(freqs)

    # Initialising the l and m vectors.
    l_vec = np.linspace(-L/2,L/2,N)
    m_vec = np.linspace(-M/2,M/2,N)

    # Defining the differential pixel sizes.
    dl = np.abs(l_vec[1]-l_vec[0])
    dm = np.abs(m_vec[1]-m_vec[0])

    ## Much cleaner way to create skymodels. Will improve with time.
    Sky_mod = Osiris.Skymodel((N,N,len(channel.fine)),l_vec,m_vec)

    freq_cent = freqs[int(len(chans)/2)]

    #
    ## Loading in Sky-model parameters.
    #

    # Default case, read in sky-model.
    col_names = ['Name','RA','u_RA','DEC','u_DEC','Sint','u_Sint','Maj','u_Maj','Min','u_Min','PA','u_PA','alpha','ModelID']

    filename = parent_path + 'CenA-GP-gauss_model.fits'

    # Loading in supernova remnant catalogue.
    SNR_data = fits.getdata(filename)
    t_SNR = Table(SNR_data)

    colnames = t_SNR.colnames

    # Decomposing the input model into columns.
    RA = np.array(t_SNR['RA']) # [deg]
    DEC = np.array(t_SNR['DEC']) # [deg]

    # Calculating the Azimuth and Dec.
    Alt,Az,Zen = Osiris.mwa_alt_az_za(obsid, RA, DEC, degrees=True) # [deg,deg,deg]

    # Subsetting sources above the horizon.
    horizon_ind = Alt > 0.0

    Alt = Alt[horizon_ind]
    Az = Az[horizon_ind]
    RA = RA[horizon_ind]
    DEC = DEC[horizon_ind]
    
    # Subsetting the table for the sky.
    t_sky = t_SNR[horizon_ind]

    #
    ## Splitting the SNRs and CenA into separate tables.
    #
    t_GP = t_sky[t_sky['Name'] != 'CenA']

    # Removing Pupis A
    if options.no_pupA:
        index_GP_vec = np.arange(len(t_GP))
        index_GP_vec = index_GP_vec[t_GP['ModelID'] != 174]
        t_GP = t_GP[t_GP['ModelID'] != 174] 
        
        name = name + 'no-PupA_'
    else:
        pass

    # Removing Vela
    if options.no_vela:
        index_GP_vec = np.arange(len(t_GP))
        index_GP_vec = index_GP_vec[t_GP['ModelID'] != 176]
        t_GP = t_GP[t_GP['ModelID'] != 176] 
        name = name + 'no-Vela_'
    else:
        pass

    # Removing Crab
    if options.no_crab:
        index_GP_vec = np.arange(len(t_GP))
        index_GP_vec = index_GP_vec[t_GP['ModelID'] != 168]
        t_GP = t_GP[t_GP['ModelID'] != 168] 

        name = name + 'no-Crab_'
    else:
        pass
        
    t_cenA = t_sky[t_sky['Name'] == 'CenA']
    
    if len(t_cenA) < 1:
        # If the table length is less than one, then CenA is below the horizon.
        print('CenA is not present in the sky-model.')
        cenA_cond = False
    else:
        print('CenA is present in the sky-model.')
        cenA_cond = True


    if (options.just_cenA and cenA_cond) and not(options.all_sky):
        # Option if you just want a sky-model of CenA
        print('Creating sky-model just with CenA...')
        
        name = name + 'just-cenA'

        Alt_cenA = Alt[t_sky['Name'] == 'CenA']
        Az_cenA = Az[t_sky['Name'] == 'CenA']

        # Loading in the model parameters.
        PA = np.array(t_cenA['PA']) + 90 # [deg]
        amaj = np.array(t_cenA['Maj'])/60 # [deg]
        bmin = np.array(t_cenA['Min'])/60 # [deg]

        # Setting the minimum size of the Gaussians:
        amaj[amaj <= np.sin(dl)] = np.sin(dl)
        bmin[bmin <= np.sin(dl)] = np.sin(dl)

        S_int = np.array(t_cenA['Sint']) # [Jy]
        SI = np.array(t_cenA['alpha']) # Spectral index.
        model_ID = np.array(t_cenA['ModelID'])

        # Calculating the flux density for each source and each fine channel.
        S_arr = np.array([S_int*(chan/200e+6)**(-SI) for chan in channel.fine]).T

        #
        ## Adding CenA Gaussian components to the sky-model.
        #

        window_size = L 
        Sky_mod.add_Gaussian_sources(Az_cenA,Alt_cenA,amaj,bmin, PA, S_arr, window_size)
        # Adjusting for the varying pixel area as a function of radius.
        #Sky_mod.model = Sky_mod.model/(np.cos(np.pi/2 - Sky_mod.Alt_grid)[:,:,None]) # Projection already accounted for.
        Sky_mod.model = Sky_mod.model # Projection already accounted for.
        
        if options.save_plots:
            # Saving the sky-model data into a compressed array.
            Sky_mod.plot_sky_mod(vmax=0.536e+6,figsize=(7,7), title = out_path + name)
        else:
            pass

        # Saving the sky-model data into a compressed array.
        np.savez_compressed(parent_path + name, Sky_mod = Sky_mod.model, N_ker_size=N_ker_size)

        sys.exit('Script Finished.')
    else:
        pass


    if (options.just_crab and not(options.no_crab)) and not(options.all_sky):
        # Option if you just want a sky-model of the Crab nebula.
        # Might need to add some message here just incase user incorrectly uses the script. 
        print('Creating sky-model just with the Crab nebula...')
        
        name = name + 'just-crab'

        Alt_crab = Alt[t_sky['ModelID'] == 168]
        Az_crab = Az[t_sky['ModelID'] == 168]

        crab_ind = t_sky['ModelID'] == 168

        # Loading in the model parameters.
        PA = np.array(t_sky['PA'][crab_ind]) + 90 # [deg]
        amaj = np.array(t_sky['Maj'][crab_ind])/60 # [deg]
        bmin = np.array(t_sky['Min'][crab_ind])/60 # [deg]
        S_int = np.array(t_sky['Sint'][crab_ind]) # [Jy]
        SI = np.array(t_sky['alpha'][crab_ind]) # Spectral index.
        model_ID = np.array(t_sky['ModelID'][crab_ind])


        # Setting the minimum size of the Gaussians:
        amaj[amaj <= np.sin(dl)] = np.degrees(dl)#180/N
        bmin[bmin <= np.sin(dl)] = np.degrees(dl)#180/N

        # Calculating the flux density for each source and each fine channel.
        S_arr = np.array([S_int*(chan/200e+6)**(-SI) for chan in channel.fine]).T

        #
        ## Adding CenA Gaussian components to the sky-model.
        #

        window_size = L 
        Sky_mod.add_Gaussian_sources(Az_crab,Alt_crab,amaj,bmin, PA, S_arr, window_size)
        # Adjusting for the varying pixel area as a function of radius.
        #Sky_mod.model = Sky_mod.model/(np.cos(np.pi/2 - Sky_mod.Alt_grid)[:,:,None]) # Projection already accounted for.
        Sky_mod.model = Sky_mod.model
        
        if options.save_plots:
            # Saving the sky-model data into a compressed array.
            Sky_mod.plot_sky_mod(vmax=0.536e+6,figsize=(7,7), title = out_path + name)
        else:
            pass

        # Saving the sky-model data into a compressed array.
        np.savez_compressed(parent_path + name, Sky_mod = Sky_mod.model, N_ker_size=N_ker_size)

        sys.exit('Script Finished.')
    else:
        pass


    if options.all_sky and not(options.just_cenA):
        
        # Option if you want the entire sky-model.
        name = name + 'all-sky_'

        if options.no_pupA or options.no_crab or options.no_vela:
            # Probably a better way of doing this.
            Alt_GP = Alt[index_GP_vec] # Removing Pupis A
            Az_GP = Az[index_GP_vec] # Removing Vela
        else:
            Alt_GP = Alt[t_sky['Name'] != 'CenA']
            Az_GP = Az[t_sky['Name'] != 'CenA']

        PA = np.array(t_GP['PA']) + 90# [deg]
        
        amaj = np.array(t_GP['Maj'])/60 # [deg]
        bmin = np.array(t_GP['Min'])/60 # [deg]
        S_int = np.array(t_GP['Sint']) # [Jy]
        SI = np.array(t_GP['alpha'])
        model_ID = np.array(t_GP['ModelID'])

        # Calculating the flux density array.
        S_arr = np.array([S_int*(chan/200e+6)**(-SI) for chan in channel.fine]).T

        # Sets get rid of multiple copies of values. 
        # Can use np.unique, but this auto sorts the array. 
        IDs_uniq = np.array(list(set(model_ID)))

        # Summing the model component fluxes for each model.
        Stot_vec = np.array([np.sum(S_int[model_ID==ID]) for ID in IDs_uniq])

        if options.save_part_mods:
            print('Option --save_partial_models chosen.')
            # Logarithmic bins characterise the data better.
            percent_vec = np.array([0,2.5e-3,1e-2,1e-1,0.9,1])

            # Index array, useful for keeping track of indices.
            index_arr = np.arange(len(model_ID))

            # Sorted from faintests to brightest models.
            sort_ind = np.argsort(Stot_vec)
            IDs_uniq = IDs_uniq[sort_ind]

            # Defining the cumulative sum vector.
            cum_sum_vec = []
            cum_sum = 0
            for Stot in Stot_vec[sort_ind]:
                cum_sum = cum_sum + Stot
                cum_sum_vec.append(cum_sum)

            # Normalising by the total flux density sum.
            cum_sum_vec = np.array(cum_sum_vec)/cum_sum

            print('Calculating partial sky model...')
            # Looping through and adding models to the sky-model.
            for i in range(len(percent_vec)-1):
                
                
                print('Sky-model total flux range: %s > cumulative sum <= %s' % (percent_vec[i],percent_vec[i+1]))
                
                temp_name = name + 'partial_leq%s' % percent_vec[i+1]

                temp_ind_arr = (cum_sum_vec > percent_vec[i])*(cum_sum_vec <= percent_vec[i+1])
                temp_ID_arr = IDs_uniq[temp_ind_arr]

                # Creating a index subset.
                index_vec = np.hstack([index_arr[model_ID == ID] for ID in temp_ID_arr])

                window_size = L 
                Sky_mod.add_Gaussian_sources(Az_GP[index_vec],Alt_GP[index_vec],amaj[index_vec],bmin[index_vec], \
                    PA[index_vec], S_arr[index_vec], window_size)
                
                # Saving the sky-model data into a compressed array.
                #np.savez_compressed(parent_path + temp_name, \
                #    Sky_mod = Sky_mod.model/(np.cos(np.pi/2 - Sky_mod.Alt_grid)[:,:,None]), N_ker_size=N_ker_size)
                np.savez_compressed(parent_path + temp_name, Sky_mod = Sky_mod.model, N_ker_size=N_ker_size) # Projection already accounted for.
                print('Saved partial model %s' % temp_name)

                if options.save_plots:
                    # Plotting and saving sky-model.
                    Sky_mod.plot_sky_mod(vmax=0.536e+6,figsize=(7,7), title = out_path + temp_name)
                else:
                    pass
        else:
            window_size = L 
            Sky_mod.add_Gaussian_sources(Az_GP,Alt_GP,amaj,bmin,PA,S_arr,window_size)
            
            temp_name = name + 'no-cenA'

            # Saving the sky-model data into a compressed array.
            #np.savez_compressed(parent_path + temp_name, \
            #    Sky_mod = Sky_mod.model/(np.cos(np.pi/2 - Sky_mod.Alt_grid)[:,:,None]), N_ker_size=N_ker_size)
            np.savez_compressed(parent_path + temp_name, Sky_mod = Sky_mod.model, N_ker_size=N_ker_size) # Projection already accounted for.
            print('Saved partial model %s' % temp_name)

            if options.save_plots:
                # Plotting and saving sky-model.
                Sky_mod.plot_sky_mod(vmax=0.536e+6,figsize=(7,7), title = out_path + temp_name)
            else:
                pass

            if not(options.no_cenA):
                print('Adding CenA to the sky-model...')
                
                Alt_cenA = Alt[t_sky['Name'] == 'CenA']
                Az_cenA = Az[t_sky['Name'] == 'CenA']

                # Option if you want the entire sky-model but without CenA.
                name = name + 'cenA'
                PA = np.array(t_cenA['PA']) + 90# [deg]
                amaj = np.array(t_cenA['Maj'])/60 # [deg]
                bmin = np.array(t_cenA['Min'])/60 # [deg]
                S_int = np.array(t_cenA['Sint']) # [Jy]
                SI = np.array(t_cenA['alpha'])

                # Calculating the flux density for each source and each fine channel.
                S_arr = np.array([S_int*(chan/200e+6)**(-SI) for chan in channel.fine]).T

                #
                ## Adding CenA Gaussian components to the sky-model.
                #

                window_size = L 
                Sky_mod.add_Gaussian_sources(Az_cenA,Alt_cenA,amaj,bmin,PA, S_arr, window_size)
                
                # Adjusting for the varying pixel area as a function of radius.
                #Sky_mod.model = Sky_mod.model/(np.cos(np.pi/2 - Sky_mod.Alt_grid)[:,:,None]) # Projection already accounted for.
                Sky_mod.model = Sky_mod.model # Projection already accounted for.
                
                # Saving the sky-model data into a compressed array.
                np.savez_compressed(parent_path + name, Sky_mod = Sky_mod.model, N_ker_size=N_ker_size)

                if options.save_plots:
                    # Plotting and saving sky-model.
                    Sky_mod.plot_sky_mod(vmax=0.536e+6,figsize=(7,7), title=out_path +name)
                else:
                    pass

    else:
        sys.exit('Selected --all_sky and --just_cenA. Not compatible.')

else:
    pass