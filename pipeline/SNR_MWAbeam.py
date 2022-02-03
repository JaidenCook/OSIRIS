#!/usr/bin/python

__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "0.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"

# Generic stuff:
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
#warnings.simplefilter('ignore', np.RankWarning)

# Parser options:
from optparse import OptionParser

# MWA beam stuff
from mwa_pb import primary_beam as pb
import mwa_hyperbeam

sys.path.append(os.path.abspath("/home/jaiden/Documents/EoR/OSIRIS"))
import Osiris

def MWA_Beam_calc(Az,Alt,freq,delays):
    """
    Returns the MWA tile beam for a range of frequencies.

            Parameters:
                    Az (numpy array): 1D flattened Azimuth numpy array. [deg]
                    Alt (numpy array): 1D flattened Altitude array. [deg]
                    freqs (list): List of MWA coarse channels. [Hz]
                    delays (list): List of MWA tile delays.

            Returns:
                    beam_cube (numpy array): Flattened array of beam values.
    """
    beam = mwa_hyperbeam.FEEBeam()

    norm_to_zenith = True
    amps = [1]*16

    Zen = np.pi/2 - Alt

    # Now we want to generate the beam for the set of Altitude and Azimuth values:
    temp_jones = beam.calc_jones_array(Az, Zen, freq, delays[0,:].astype('int'), amps, norm_to_zenith)
    
    xx_temp = temp_jones[:,0]*np.conjugate(temp_jones[:,0]) + temp_jones[:,1]*np.conjugate(temp_jones[:,1])
    yy_temp = temp_jones[:,2]*np.conjugate(temp_jones[:,2]) + temp_jones[:,3]*np.conjugate(temp_jones[:,3])

    return np.abs(xx_temp + yy_temp)/2

def MWA_Beam_vec(Az_vec,Alt_vec,freqs,delays):
    """
    Returns the MWA tile beam for a range of frequencies.

            Parameters:
                    Az_arr (numpy array): 2D Azimuth numpy array. [deg]
                    Alt_arr (numpy array): 2D Altitude array. [deg]
                    ind_arr (numpy array): 2D index array, all Alt/Az points with r <= 1.
                    freqs (list): List of MWA coarse channels. [Hz]
                    delays (list): List of MWA tile delays.

            Returns:
                    beam_cube (numpy array): 3D numpy array, contains the beam value for each
                    Alt/Az for each frequency.
    """

    beam = mwa_hyperbeam.FEEBeam()

    Beam_arr = np.zeros((len(Az_vec),len(freqs)))

    for i in range(len(freqs)):

        Beam_arr[:,i] = MWA_Beam_calc(Az_vec,Alt_vec,freqs[i],delays)

    return Beam_arr


def MWA_beam(Az_arr, Alt_arr, ind_arr, freqs, delays, interp_cond=False):
    """
    Returns the MWA tile beam for a range of frequencies.

            Parameters:
                    Az_arr (numpy array): 2D Azimuth numpy array. [deg]
                    Alt_arr (numpy array): 2D Altitude array. [deg]
                    ind_arr (numpy array): 2D index array, all Alt/Az points with r <= 1.
                    freqs (list): List of MWA coarse channels. [Hz]
                    delays (list): List of MWA tile delays.

            Returns:
                    beam_cube (numpy array): 3D numpy array, contains the beam value for each
                    Alt/Az for each frequency.
    """

    # Initialising the beam array:
    beam_cube = np.zeros([len(Az_arr),len(Az_arr),len(freqs)]) # Might not need to keep the beam cube.

    delays = np.array([delays,delays])

    print('Generating the MWA primary beam for each coarse channel.')
    
    # Creating MWA beam object.
    beam = mwa_hyperbeam.FEEBeam()

    # Hyperbeam options.
    amps = [1.0] * 16
    norm_to_zenith = True
    #norm_to_zenith = False

    start = time.perf_counter()

    Zen_temp = np.pi/2 - Alt_arr

    if interp_cond:
        print('Interpolating...')
        # interpolate as a function of frequency.
        chans_coarse = np.arange(115,151) # Coarse channels 115 to 150 endpoint not included.
        freq_coarse = chans_coarse*1.28e+6 #[Hz]

        print('Length of coarse channel grid = %s' % len(chans_coarse))

        beam_coarse_cube = np.zeros([len(Az_arr),len(Az_arr),len(freq_coarse)])

        for i in range(len(freq_coarse)):

            beam_coarse_cube[ind_arr,i] = MWA_Beam_calc(Az_arr[ind_arr],Alt_arr[ind_arr],freq_coarse[i],delays)

        # Interpolating:
        c = 299792458.0 #[m/s]
        waves_coarse = c/freq_coarse

        from scipy.interpolate import interp1d

        Beam_interp = interp1d(waves_coarse, beam_coarse_cube, kind='cubic')

        # Calculating the beam for each fine channel.
        lam_fine_chans = c/freqs #[m]
        beam_cube = Beam_interp(lam_fine_chans)

    else:
        for i in range(len(freqs)):
            # Default no interpolation, nearest neighbour.
            beam_cube[ind_arr,i] = MWA_Beam_calc(Az_arr[ind_arr],Alt_arr[ind_arr],freqs[i],delays)

    end = time.perf_counter()
    print('Beam generation time = %6.3f s' %  (end - start))

    return beam_cube


class channels:
    """
    Calculates the vector of fine channels. The outputs additionally include the fine channel flag indices,
    and the beam indices. All fine channels in a particular coarse channel are assumed to have the same 
    beam value.

    ...

    Attributes
    ----------
    fine : numpy array
        1D array of fine channel frequencies in Hz.
    beam_inds : str
        Coarse channel index for each fine channel.
    flag_inds : int
        Index of flagged fine channels.
    chan_inds : int
        Index array for all fine channels.
    chan_flag_inds : int
        Index array for the unflagged fine channels.
    bandiwdth : float
        Total bandwidth spanned by the fine channels.

    Methods
    -------
    calc_fine_chans(freqs=""):
        Calculate the fine channels, the flag indices, and the beam indices for input coarse channels freqs.
    """

    # Constants:
    fine_chan_width = 0.08e+6 # [Hz]
    coarse_chan_width = 1.28e+6 # [Hz]
    N_fine_chans = int(coarse_chan_width/fine_chan_width)

    def __init__(self,freqs,N_fine_chans=N_fine_chans):
        """
        Initialised the channel object attributes.

        Parameters
        ----------
            freqs : numpy array
                1D numpy array of fine channel frequencies in Hz.
            N_fine_chans : int
                Number of fine channels in a coarse channel. 
        """

        # Initialising empty arrays.
        self.fine = np.array([])# Empty fine channel array. 
        self.beam_inds = np.array([]) # Creating a coarse list of frequency values for beam generation.

        # Definfing middle and last fine channels. Generalised.
        chan_mid = int(N_fine_chans/2)-1
        chan_end = N_fine_chans-1

        self.flag_inds = np.array([0,chan_mid,chan_end]) # Flagging the first, middle and last fine channel.
        self.chan_inds = np.arange(N_fine_chans*len(freqs)) # Fine channel index vector used for masking.
        
    
    def calc_fine_chans(self,freqs,N_fine_chans=N_fine_chans,\
        coarse_chan_width=coarse_chan_width,fine_chan_width=fine_chan_width):
        """
        Calculates the vector of fine channels. The outputs additionally include the fine channel flag indices,
        and the beam indices. All fine channels in a particular coarse channel are assumed to have the same 
        beam value.

        Parameters
        ----------
        freqs : numpy array
            1D numpy array of fine channel frequencies in Hz.
        N_fine_chans : int
            Number of fine channels in a coarse channel. 
        coarse_chan_width : float
            Coarse channel width in Hz.

        Returns
        -------
        None
        """

        # Definfing middle and last fine channels. Generalised.
        chan_mid = int(N_fine_chans/2)-1
        chan_end = N_fine_chans-1

        # Looping through each coarse channel.
        i = 0
        for freq in freqs:

            # Determining the low and high frequnecies for the particular coarse channel.
            coarse_low = freq - coarse_chan_width/2
            coarse_hi = freq + coarse_chan_width/2

            # Determining the fine channels for this particulay coarse channel.
            #temp_fine_chans = np.linspace(coarse_low, coarse_hi, N_fine_chans)
            # Defined at the centre.
            temp_fine_chans = coarse_low + fine_chan_width/2 + fine_chan_width*np.arange(N_fine_chans)

            if i > 0:
                # Creating the index of flagged fine channels.
                self.flag_inds = np.concatenate((self.flag_inds,\
                    np.array([0 + i*N_fine_chans, chan_mid + i*N_fine_chans, chan_end + i*N_fine_chans])))

            # There is a more efficient way to do this, but this isn't too time or memory consuming.
            # Concatenating the list of fine channels to array.
            self.fine = np.concatenate((self.fine,temp_fine_chans))
            self.beam_inds = np.concatenate((self.beam_inds,np.ones(N_fine_chans)*i))

            i += 1

        # We separately want to specify a channel index array for the non-flagged channels.
        zeros_vec = np.zeros(len(self.chan_inds))
        zeros_vec[self.flag_inds] = self.flag_inds # Vector for masking purposes.

        # Here I am masking the flagged channels.
        # Creating a new index vector of the unflagged channels.
        self.chan_flag_inds = (self.chan_inds - zeros_vec).astype(int)
        self.chan_flag_inds = self.chan_flag_inds[self.chan_flag_inds > 0]

        # Subsetting the beam channel index vector. Can use these to attenuate the sky-model by the 
        # appropriate beam map.
        self.beam_inds = self.beam_inds[self.chan_flag_inds]

        # Calculating the total bandwidth.
        self.bandwidth = (self.fine[len(self.fine)-1] - self.fine[0])/1e+6 + fine_chan_width/1e+6
        self.finewifth = fine_chan_width/1e+6
