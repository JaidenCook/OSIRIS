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
        chans_coarse = np.arange(115,157) # Coarse channels 115 to 155 endpoint not included.
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

def beam_kernel(freq_vec,N_ker_size,L,M,delays,gauss_beam=False,interp_cond=False):
    """
    For an input set of frequencies, calculate the beam image.
    This is used to construct the visibility degridding kernel.

    Parameters
        ----------
        freq_vec : numpy array, float
            Vector of fine channels in Hz. 
        N_ker_size : int
            Size of the beam kernel. This should be an odd number.
        L : float
            Size of the image space array in the l-direction.
        M : float
            Size of the image space array in the m-direction.

        Returns
        -------
        Beam cube.
    """
    
    ### Might be better being incorporated into the kernel object.

    l_b_vec = np.linspace(-L/2,L/2,N_ker_size)
    m_b_vec = np.linspace(-M/2,M/2,N_ker_size)

    # Creating the grid:
    l_b_arr, m_b_arr = np.meshgrid(l_b_vec, m_b_vec)
    # Why the hell do I do this? Is this back to front?
    m_b_arr = m_b_arr # Should probably fix this issue with the sky-model class.

    # Creating a radius array for masking purposes:
    r_b_arr = np.sqrt(l_b_arr**2 + m_b_arr**2)

    # Creating an index array, we want all pixels less than or equal to r = 1:
    ind_b_arr = r_b_arr <= 1.0

    # Here we want to create a new alt and az array that is the same size as l_arr and m_arr:
    Alt_b_arr = np.zeros(np.shape(l_b_arr))
    Az_b_arr = np.zeros(np.shape(l_b_arr))

    # Now we want to determine the Altitude and Azimuth, but only in the region where r <= 1. Outside this region is 
    # beyond the boundary of the horizon.
    # Alt = arccos([l^2 + m^2]^(1/2))
    Alt_b_arr[ind_b_arr] = np.arccos(r_b_arr[ind_b_arr]) 
    #arctan2() returns [-pi,pi] we want [0,2pi].
    Az_b_arr[ind_b_arr] = 2*np.pi - (np.arctan2(l_b_arr[ind_b_arr],-m_b_arr[ind_b_arr]) + np.pi) 

    if gauss_beam:
        # For testing purposes. Use a Gaussian beam.
        X = np.arange(len(l_b_arr))
        Y = np.arange(len(l_b_arr))

        xx,yy = np.meshgrid(X,Y)

        x_cent = 0.5*(np.max(X) - np.min(X))
        y_cent = 0.5*(np.max(X) - np.min(X))

        sigx = 2
        sigy = 2

        amaj = sigx * (2.0*np.sqrt(2.0*np.log(2.0)))
        bmin = sigy * (2.0*np.sqrt(2.0*np.log(2.0)))

        Amplitude = 2*np.pi*sigx*sigy

        beam_temp = Osiris.Gauss2D(xx, yy, Amplitude, x_cent, y_cent, 0, amaj, bmin)

        Beam_cube = np.ones((len(l_b_arr),len(m_b_arr),len(freq_vec)))*beam_temp[:,:,None]

    else:
        # Need to interpolate the beam for the fine channels.
        # Calculating the FEE beam values for each coarse channel.
        Beam_cube = MWA_beam(Az_b_arr,Alt_b_arr,ind_b_arr,freq_vec,delays,interp_cond=interp_cond)

        plot_cond = False # Testing the output primary beam is orientated the correct way.
        if plot_cond:
            import matplotlib.pyplot as plt
            #Osiris.Plot_img(Beam_cube[:,:,0],projection='polar',X_vec=Az_b_arr,Y_vec=np.cos(Alt_b_arr))
            Osiris.Plot_img(Beam_cube[:,:,0],X_vec=l_b_arr,Y_vec=m_b_arr)
            plt.show()
        else:
            pass
    
    #
    ## Testing the beam solid angle.
    #
    dl = np.abs(l_b_vec[1]-l_b_vec[0])
    dm = np.abs(m_b_vec[1]-m_b_vec[0])
    dA = dl*dm

    # Calculating the solid angle for the low and high band beam kernels.
    sky_solid_angle_low = np.sum(dA/(np.sin(Alt_b_arr[Beam_cube[:,:,0] >= 0.5])))
    sky_solid_angle_hi = np.sum(dA/(np.sin(Alt_b_arr[Beam_cube[:,:,-1] >= 0.5])))

    # Calculating the average.
    sky_solid_angle_avg = 0.5*(sky_solid_angle_hi + sky_solid_angle_low)

    # Calculating the grid kernel size.
    sig_u = np.sqrt((2*np.log(2))/(np.pi*sky_solid_angle_avg)) # lambda

    print('Approximate beam low solid angle = %5.4f [Sr]' % (sky_solid_angle_low))
    print('Approximate beam high solid angle = %5.4f [Sr]' % (sky_solid_angle_hi))
    print('Approximate beam avg solid angle = %5.4f [Sr]' % (sky_solid_angle_avg))

    return Beam_cube, l_b_arr, m_b_arr, sig_u

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

        # Assigning the coarse channels to self.
        self.coarse = freqs
        
    
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
        self.bandwidth = (self.fine[len(self.fine)-1] - self.fine[0])/1e+6 + fine_chan_width/1e+6 #[MHz]
        self.finewidth = fine_chan_width/1e+6 # [MHz]
