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

sys.path.append(os.path.abspath("/home/jaiden/Documents/EoR/OSIRIS"))
import Iris
from Iris_degrid import *
from Iris_grid import *
import SNR_MWAbeam
import SNR_21cmps

def Vis_noise(freq_vec,dt,fine_chan_width=0.08e+6,Trec=50,Aeff=21.5):
    """
    For an input set of frequencies, and time in seconds, calculate
    the visibility noise using the radiometer equation. default MWA
    values are given.

    Parameters
        ----------
        freq_vec : numpy array, float
            Vector of fine channels in Hz. 
        dt : float
            Observation time in seconds

        Returns
        -------
        Vector of sigma values for each fine channel.
    """

    # Boltzmans constant
    kb = 1380.648 #[Jy K^-1 m^2]

    freq_temp_vec = freq_vec/1e+6 # [MHz]

    # calculate sky temperature.
    Tsky_vec = 228*(freq_temp_vec/150)**(-2.53)

    # Standard deviation term for the noise:
    sigma_vec = 2*(kb/Aeff)*(Tsky_vec + Trec)*(1/np.sqrt(fine_chan_width*dt)) #[Jy]

    return sigma_vec

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

        beam_temp = Iris.Gauss2D(xx, yy, Amplitude, x_cent, y_cent, 0, amaj, bmin)

        Beam_cube = np.ones((len(l_b_arr),len(m_b_arr),len(freq_vec)))*beam_temp[:,:,None]

    else:
        # Need to interpolate the beam for the fine channels.
        # Calculating the FEE beam values for each coarse channel.
        Beam_cube = SNR_MWAbeam.MWA_beam(Az_b_arr,Alt_b_arr,ind_b_arr,freq_vec,delays,interp_cond=interp_cond)

    return Beam_cube, l_b_arr, m_b_arr

def calc_powerspec(Sky_mod,MWA_array,channel,u_lam_arr,v_lam_arr,freq_cent,\
    constants,delays,add_21cm=False,gauss_beam=False,interp_cond=False,\
    wedge_cond=False,cosmo=None,beam_cond=True,taper_cond=True,wproj_cond=True):
    """
    calculate the 1D and 2D power spectrums for an input sky-model, with an input
    interferometric array and channel layout.

    Parameters
        ----------
        Sky_mod : object, float
            stuff
        MWA_array : object, float
            stuff
        channel : object, float
            stuff
        u_lam_arr : numpy array, float
            2D numpy array of u coordinates in wavelengths.
        v_lam_arr : numpy array, float
            2D numpy array of u coordinates in wavelengths.
        freq_cent : float
            Central frequency of the observing band in Hz.
        constants : list
            List of constants used for calculations [L,M,N,N_ker_size,uvmax,c].
        delays : numpy array, int
            Array of delays used for constructing the MWA primary beam.

        Returns
        -------
        Power : object, float
    """

    np.random.seed(0)

    # Unpacking constants.
    L = constants[0]
    M = constants[1]
    N = constants[2] # Might not need this.
    N_ker_size = constants[3]
    uvmax = constants[4] # [lambda]
    c = constants[5] # Speed of light. [m/s]

    try:
        # This is a filthy temporary fix. Earlier versions of the pipeline don't 
        # have the Tobs_hours parameter. This sets a default based on the try statement.
        if constants[6]:
            Tobs_hours = constants[6]

    except IndexError:
            Tobs_hours = 10

    dA = (L*M)/(N**2) # Area element.

    # Better this way.
    u_lam_vec = u_lam_arr[0,:]
    v_lam_vec = v_lam_arr[:,0]

    #
    ## Calculating the True visibilities
    #   

    # Initialising the complex visibility cube.
    Vis_cube = np.zeros(np.shape(Sky_mod.model),dtype=complex)

    print('Computing sky visibilities for each un-flagged fine channel...')
    #for i in channel.chan_flag_inds:
    for i in np.arange(len(channel.fine)):

        Vis_cube[:,:,i] = Iris.Visibilities_2D(Sky_mod.model[:,:,i])*dA

    #
    ## Calculating kernel beam cube
    #    

    if beam_cond:
        # Default condition, generate the beam.
        beam_cube, l_b_arr, m_b_arr = beam_kernel(channel.fine,N_ker_size,L,M,delays,\
            gauss_beam=gauss_beam,interp_cond=interp_cond)
        print('Delays : %s' % (np.array(delays).astype('int')))
    else:
        # If there is no beam, then the beam kernel is uniform.
        # This still includes w-projection.

        print('No primary beam...')

        l_b_vec = np.linspace(-L/2,L/2,N_ker_size)
        m_b_vec = np.linspace(-M/2,M/2,N_ker_size)

        # Creating the grid:
        l_b_arr, m_b_arr = np.meshgrid(l_b_vec, m_b_vec)
        m_b_arr = m_b_arr # Should probably fix this issue with the sky-model class.

        #beam_cube = np.ones((len(l_b_arr),len(m_b_arr),len(channel.fine)))
        beam_cube = np.zeros((len(l_b_arr),len(m_b_arr),len(channel.fine)))

        # Cube of all ones has non-zero weights below the horizon. Causes huge sidelobes. Not good.
        beam_cube[int(N_ker_size/2),int(N_ker_size/2),:] = 1
        

    #
    ## Calculating the noise standard devitions
    # 

    Tobs_sec = Tobs_hours*3600 #[seconds]
    sigma_vec = Vis_noise(channel.fine,Tobs_sec,fine_chan_width=0.08e+6,Trec=50,Aeff=21.5)

    #
    ## Degridding
    #    

    lam_fine_chans = c/channel.fine

    print('Degridding and gridding...')

    if wproj_cond:
        print('Performing w-projection.')
    else:
        print('Not performing w-projection.')

    u_lam_list = []
    v_lam_list = []
    vis_list = []
    baseline_list = []

    counter = 0
    #for i in channel.chan_flag_inds:
    for i in np.arange(len(channel.fine)):# Not flagging.

        kernel = w_kernel(beam_cube[:,:,i],l_b_arr,m_b_arr)

        ## Defining the temporary u,v, and w vectors for the ith fine channel.
        # The (u,v,w) values for the ith fine channel:
        MWA_array.uvw_lam(lam_fine_chans[i],uvmax)
        u_lam_temp = MWA_array.u_lam
        v_lam_temp = MWA_array.v_lam
        w_lam_temp = MWA_array.w_lam

        if wproj_cond:
            # Calculating the degridded sky visibilities.
            Vis_sky_deg = Vis_degrid(kernel,u_lam_vec,v_lam_vec,u_lam_temp,v_lam_temp,Vis_cube[:,:,i],
                                    w_lam_temp,phase_cond=True)
        else:
            # Calculating the degridded sky visibilities.
            Vis_sky_deg = Vis_degrid(kernel,u_lam_vec,v_lam_vec,u_lam_temp,v_lam_temp,Vis_cube[:,:,i],
                                    phase_cond=True)

        ## There are negative pairs due to the hermitian nature of visibilities.
        u_lam_list.append(np.concatenate((MWA_array.u_lam,-MWA_array.u_lam)))
        v_lam_list.append(np.concatenate((MWA_array.v_lam,-MWA_array.v_lam)))

        # Determining the complex conjugate values.
        Vis_sky_deg = np.concatenate((Vis_sky_deg,np.conjugate(Vis_sky_deg)))

        vis_list.append(Vis_sky_deg)
        baseline_list.append(len(Vis_sky_deg))

        # Adding white Gaussian noise:
        Vis_noise_real = np.ones(len(Vis_sky_deg))*\
         np.random.normal(0.0, sigma_vec[i], len(Vis_sky_deg))/np.sqrt(2)

        Vis_noise_imag = np.ones(len(Vis_sky_deg))*\
         np.random.normal(0.0, sigma_vec[i], len(Vis_sky_deg))/np.sqrt(2)

        # Adding noise.
        Vis_sky_deg.real = Vis_sky_deg.real + Vis_noise_real
        Vis_sky_deg.imag = Vis_sky_deg.imag + Vis_noise_imag

        counter += 1

    #
    ## Gridding
    #    

    # No flagged fine channels for gridding purposes.
    #temp_ind = channel.chan_flag_inds

    # Initialising the gridded visibility cube:
    gridded_vis_cube = np.zeros(np.shape(Sky_mod.model),dtype=complex)
    vis_weights_cube = np.zeros((N,N,len(channel.fine)))

    # Natural weighting
    #gridded_vis_cube[:,:,temp_ind], vis_weights_cube[:,:,temp_ind] = grid_cube(u_lam_list,v_lam_list,\
    #    vis_list, u_lam_arr, v_lam_arr,gridded_vis_cube[:,:,temp_ind], vis_weights_cube[:,:,temp_ind],\
    #        weighting='natural')

    # Gaussian weighting
    #gridded_vis_cube[:,:,temp_ind], vis_weights_cube[:,:,temp_ind] = grid_cube(u_lam_list,v_lam_list,\
    #        vis_list, u_lam_arr, v_lam_arr,gridded_vis_cube[:,:,temp_ind], vis_weights_cube[:,:,temp_ind],\
    #            weighting='gaussian')
    
    print('Sky-model shape')
    print(Sky_mod.model.shape)

    # Removing the fine channel flagging. Use idealised scenario.
    gridded_vis_cube, vis_weights_cube = grid_cube(u_lam_list,v_lam_list,\
            vis_list, u_lam_arr, v_lam_arr,gridded_vis_cube, vis_weights_cube,\
                weighting='gaussian')

    # Applying the Blackman-Harris taper. This is applied along the freqency axis.
    #gridded_vis_cube *= signal.blackmanharris(len(channel.fine))

    if taper_cond:
        print('Applying Blackman-Harris taper.')
        gridded_vis_cube *= signal.blackmanharris(len(channel.fine))
    else:
        print('Not applying Blackman-Harris taper.')
        # When testing spectral leakage without the beam, we want to remove the Blackman Harris window.
        pass

    #
    ## Fourier transforming with respect to frequency
    #    

    # Determining the eta vector. Paramaterise this later.
    N_chans = len(channel.fine)

    # Fourier resolution:
    dnu_fine = channel.bandwidth/N_chans

    print('Fine Channel size = %5.3e [MHz]' % dnu_fine)

    # Generating the Fourier sky cube.
    Four_sky_cube = ifftshift(fftn(fftshift(gridded_vis_cube,axes=(2,)),axes=2,norm='forward'),axes=(2,))

    Four_sky_cube *= (dnu_fine*1e+6)

    eta = fftshift(fftfreq(N_chans,2*channel.bandwidth*1e+6/N_chans)) # Fix this, parameterise. Converting to Hz

    test_cond = False
    if test_cond:
        print('Saving Fourier-sky cube for analysis.')
        path = '/home/jaiden/Documents/EoR/SNR-Pipeline/workbooks/data/'
        test_name1 = 'Zen-Four-sky-cube'
        test_name2 = 'Gridded-Vis-cube'
        np.savez_compressed(path + test_name1, Four_sky_cube = Four_sky_cube, eta=eta, u_lam_arr=u_lam_arr, v_lam_arr=v_lam_arr)
        np.savez_compressed(path + test_name2, gridded_vis_cube = gridded_vis_cube, weights_cube=vis_weights_cube)
    else:
        pass

    # Subsetting out the negative eta values:
    Four_sky_cube = Four_sky_cube[:,:,eta >= 0]
    eta = eta[eta >= 0]

    if add_21cm:
        # Condition for adding in the 21cm signal.
        print('Calculating the 21cm signal visibilities...')
        Vis_21cm = SNR_21cmps.calc_21cm_pspec(freq_cent,channel,u_lam_arr,v_lam_arr,eta)
        Four_sky_cube = Four_sky_cube + Vis_21cm
    else:
        pass

    temp_avg = np.mean(vis_weights_cube, axis=2)

    # Defining the weights cube
    weights_cube = np.ones(np.shape(Four_sky_cube))*temp_avg[:,:,None].real

    del temp_avg

    # Fourier resolution:
    dnu_fine = channel.bandwidth/N_chans
    dnu = channel.bandwidth
    
    #
    ## Calculating the 1D and 2D power spectra.
    #    

    # Initialising the power spectrum object.
    Power = Iris.Power_spec(Four_sky_cube,eta,u_lam_arr,v_lam_arr,freq_cent,\
        dnu,dnu_fine,weights_cube=weights_cube,cosmo=cosmo)

    # Calculating the power spectra.
    print('Calculating the spherically average 1D power spectrum...')
    Power.Spherical(wedge_cond=wedge_cond)
    print('Calculating the cylindrically averaged 2D power spectrum...')
    Power.Cylindrical()
    
    return Power

def calc_noise_powerspec(MWA_array,channel,u_lam_arr,v_lam_arr,freq_cent,\
    constants,add_21cm=False,wedge_cond=False,cosmo=None):
    """
    calculate the 1D and 2D power spectrums for a noise like visibilities, with an input
    interferometric array and channel layout. Has the option of including the 21cm signal.

    Parameters
        ----------
        MWA_array : object, float
            stuff
        channel : object, float
            stuff
        u_lam_arr : numpy array, float
            2D numpy array of u coordinates in wavelengths.
        v_lam_arr : numpy array, float
            2D numpy array of u coordinates in wavelengths.
        freq_cent : float
            Central frequency of the observing band in Hz.
        constants : list
            List of constants used for calculations [L,M,N,N_ker_size,uvmax,c].
       
        Returns
        -------
        Power : object, float
    """

    np.random.seed(0)

    # Unpacking constants.
    L = constants[0]
    M = constants[1]
    N = constants[2] # Might not need this.
    uvmax = constants[4] # [lambda]
    c = constants[5] # Speed of light. [m/s]

    try:
        # This is a filthy temporary fix. Earlier versions of the pipeline don't 
        # have the Tobs_hours parameter. This sets a default based on the try statement.
        if constants[6]:
            Tobs_hours = constants[6]

    except IndexError:
            Tobs_hours = 10

    #
    ## Calculating the noise standard devitions
    # 

    Tobs_sec = Tobs_hours*3600 #[seconds]
    sigma_vec = Vis_noise(channel.fine,Tobs_sec,fine_chan_width=0.08e+6,Trec=50,Aeff=21.5)

    #
    ## Degridding
    #    

    lam_fine_chans = c/channel.fine

    print('Degridding and gridding...')

    u_lam_list = []
    v_lam_list = []
    vis_list = []
    baseline_list = []

    counter = 0
    #for i in channel.chan_flag_inds:
    for i in np.arange(len(channel.fine)):# Not flagging.

        ## Defining the temporary u,v, and w vectors for the ith fine channel.
        # The (u,v,w) values for the ith fine channel:
        MWA_array.uvw_lam(lam_fine_chans[i],uvmax)
        u_lam_temp = MWA_array.u_lam
        #v_lam_temp = MWA_array.v_lam
        #w_lam_temp = MWA_array.w_lam

        ## There are negative pairs due to the hermitian nature of visibilities.
        u_lam_list.append(np.concatenate((MWA_array.u_lam,-MWA_array.u_lam)))
        v_lam_list.append(np.concatenate((MWA_array.v_lam,-MWA_array.v_lam)))

        # Determining the complex conjugate values.
        Vis_sky_deg = np.zeros(u_lam_temp.shape, dtype=complex)

        # Adding white Gaussian noise:
        Vis_sky_deg.real = np.ones(len(Vis_sky_deg))*\
         np.random.normal(0.0, sigma_vec[i], len(Vis_sky_deg))/np.sqrt(2)

        Vis_sky_deg.imag = np.ones(len(Vis_sky_deg))*\
         np.random.normal(0.0, sigma_vec[i], len(Vis_sky_deg))/np.sqrt(2)

        vis_list.append(Vis_sky_deg)
        baseline_list.append(len(Vis_sky_deg))

        counter += 1

    #
    ## Gridding
    #    

    # No flagged fine channels for gridding purposes.
    #temp_ind = channel.chan_flag_inds

    # Initialising the gridded visibility cube:
    gridded_vis_cube = np.zeros((N,N,len(channel.fine)),dtype=complex)
    vis_weights_cube = np.zeros((N,N,len(channel.fine)))

    # Natural weighting.
    #gridded_vis_cube[:,:,temp_ind], vis_weights_cube[:,:,temp_ind] = grid_cube(u_lam_list,v_lam_list,\
    #    vis_list, u_lam_arr, v_lam_arr,gridded_vis_cube[:,:,temp_ind], vis_weights_cube[:,:,temp_ind],\
    #        weighting='natural')

    # Gaussian weighting.
    #gridded_vis_cube[:,:,temp_ind], vis_weights_cube[:,:,temp_ind] = grid_cube(u_lam_list,v_lam_list,\
    #    vis_list, u_lam_arr, v_lam_arr,gridded_vis_cube[:,:,temp_ind], vis_weights_cube[:,:,temp_ind],\
    #        weighting='gaussian')

    # Gaussian weighting.
    gridded_vis_cube, vis_weights_cube = grid_cube(u_lam_list,v_lam_list,\
        vis_list, u_lam_arr, v_lam_arr,gridded_vis_cube, vis_weights_cube,\
            weighting='gaussian')

    # Applying the Blackman-Harris taper. This is applied along the freqency axis.
    gridded_vis_cube *= signal.blackmanharris(len(channel.fine))

    #
    ## Fourier transforming with respect to frequency
    #    

    # Determining the eta vector. Paramaterise this later.
    N_chans = len(channel.fine)

    # Fourier resolution:
    dnu_fine = channel.bandwidth/N_chans

    print('Fine Channel size = %5.3e [MHz]' % dnu_fine)

    # Generating the Fourier sky cube.
    Four_sky_cube = ifftshift(fftn(fftshift(gridded_vis_cube,axes=(2,)),axes=2,norm='forward'),axes=(2,))

    Four_sky_cube *= (dnu_fine*1e+6)

    eta = fftshift(fftfreq(N_chans,2*channel.bandwidth*1e+6/N_chans)) # Fix this, parameterise. Converting to Hz

    # Subsetting out the negative eta values:
    Four_sky_cube = Four_sky_cube[:,:,eta >= 0]

    eta = eta[eta >= 0]

    if add_21cm:
        # Condition for adding in the 21cm signal.
        print('Calculating the 21cm signal visibilities...')
        Vis_21cm = SNR_21cmps.calc_21cm_pspec(freq_cent,channel,u_lam_arr,v_lam_arr,eta)
        Four_sky_cube = Four_sky_cube + Vis_21cm
    else:
        pass

    temp_avg = np.mean(vis_weights_cube, axis=2)

    # Defining the weights cube
    weights_cube = np.ones(np.shape(Four_sky_cube))*temp_avg[:,:,None].real

    del temp_avg

    # Fourier resolution:
    dnu_fine = channel.bandwidth/N_chans
    dnu = channel.bandwidth
    
    #
    ## Calculating the 1D and 2D power spectra.
    #    

    # Initialising the power spectrum object.
    Power = Iris.Power_spec(Four_sky_cube,eta,u_lam_arr,v_lam_arr,freq_cent,\
        dnu,dnu_fine,weights_cube=weights_cube,cosmo=cosmo)

    # Calculating the power spectra.
    print('Calculating the spherically average 1D power spectrum...')
    Power.Spherical(wedge_cond=wedge_cond)
    print('Calculating the cylindrically averaged 2D power spectrum...')
    Power.Cylindrical()
    
    return Power
