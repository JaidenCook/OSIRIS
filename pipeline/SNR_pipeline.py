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

from traitlets.traitlets import default
warnings.filterwarnings("ignore")

# Array stuff:
import numpy as np
#warnings.simplefilter('ignore', np.RankWarning)

# Plotting stuff:
import matplotlib.pyplot as plt
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
import Iris

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

def setup_log(name,log_file):

    import logging
    import sys

    file_handler = logging.FileHandler(filename=log_file)
    #file_handler.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler(sys.stdout)
    #stdout_handler.setLevel(logging.WARNING)
    handlers = [file_handler, stdout_handler]

    logging.basicConfig( level=logging.INFO,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers)

    logger = logging.getLogger(name)

    return logger

if __name__ == "__main__":

    # Defining the parser:
    usage="Usage: %prog [options]\n"
    parser = OptionParser(usage=usage)
    parser.add_option('--obsid',dest="obsid",default=None,help="Input the observation obsid.\n")
    parser.add_option('--model',dest="model",default=None,help="Input FITS file model name.\n")
    parser.add_option('--outname',dest="out",default=None,help="Output model name.\n")
    parser.add_option('--zenith_pointing',dest="zenith_pointing",default=False,action='store_true',help="If true then use zenith pointed MWA beam.\n")
    parser.add_option('--point_source',dest="point_source",default=False,action='store_true',help="If true then use a 1Jy point source for testing purposes.\n")
    parser.add_option('--flux',dest="flux",default=1,help="Flux density of test source, only works for a single source.\n")
    parser.add_option('--add_21cm',dest="add_21cm",default=False,action='store_true',help='Add 21cm signal to the visibilties.\n')
    parser.add_option('--gauss_beam',dest="gauss_beam",default=False,action='store_true',help='Use a dummy Gaussian primary beam.\n')
    parser.add_option('--beam_interp',dest="beam_interp",default=False,action='store_true',help='Interpolate FEE beam fine channels.\n')
    parser.add_option('--no_beam',dest="no_beam",default=True,action='store_false',help='Turn off the MWA primary beam.\n')
    parser.add_option('--no_spec_taper',dest="no_spec_taper",default=True,action='store_false',help='Turn off the Blackman-Harris spectral tapering.\n')
    parser.add_option('--no_wproj',dest="no_wproj",default=True,action='store_false',help='Turn off w-projection.\n')
    parser.add_option('--gaussian_source',dest="gauss_source",default=False,action='store_true',help="If true then use a 1Jy 1deg x 1deg Gaussian source for testing purposes.\n")
    parser.add_option('--plot_allsky',dest="plot_allsky",default=False,action='store_true',help="If true make a plot of the sky model.\n")
    parser.add_option('--plot_pspec',dest="plot_pspec",default=False,action='store_true',help="If true make a plot of the 1D and 2D power spectrums.\n")
    parser.add_option('--noise_only',dest="noise_only",default=False,action='store_true',help="If true only use noise and 21cm signal to construct visibilities.\n")
    parser.add_option('--no_wedge',dest="wedge_cond",default=False,action='store_true',help="If true only avrage the visibilities that aren't in the wedge.\n")
    parser.add_option('--az_source',dest="az_mod",default=None,help="Azimuth of test point/gaussian source.\n")
    parser.add_option('--alt_source',dest="alt_mod",default=None,help="Altitude of test point/gaussian source.\n")
    parser.add_option('--sky_model',dest="sky_model",default=None,help="Optionally load in a pre-made sky model.\n")
    parser.add_option('--gauss_array',dest="gauss_array",default=False,action='store_true',help="Use a Gaussian distributed array instead of the MWA Phase I array. For testing.\n")
    parser.add_option('--uniform_array',dest="uniform_array",default=False,action='store_true',help="Use a uniform distributed array instead of the MWA Phase I array. For testing.\n")
    parser.add_option('--save_sky_model',dest="save_sky_model",default=False,action='store_true',help="Save sky-model. Doesn't not work with --sky_model.\n")
    parser.add_option('--CHIPS_cosmo',dest="CHIPS_cosmo",default=False,action='store_true',help="Use the CHIPs cosmology instead of the default Plank 2018 cosmology results.\n")
    parser.add_option('--obs_length',dest="Tobs",default=10,help="Observing length, default value is 10hours. Input value in hours.\n")
    parser.add_option('--kernel_size',dest="N_ker_size",default=31,help="Degridding kernel size, default is 31 pixels.\n")

    (options, args) = parser.parse_args()

    # Initialising logger. Fill in log info later.
    log_file_name = 'snr-pipeline-{0}.log'.format(int(time.time()))

    logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.INFO,
        filename=log_file_name, filemode='w', force=True)

    logger = logging.getLogger('snrpspec')

    start0 = time.perf_counter()

    # paths:
    parent_path = '/home/jaiden/Documents/EoR/SNR-Pipeline/'

    ## Constants
    # Natural Constants:
    c = 299792458.0 #[m/s]
    MWA_lat = -26.7033194444 # [deg]

    # Sky model array dimensions.
    L = 2
    M = 2
    logger.info('Image plane (l,m) sizes: L = %s, M = %s' % (L,M))

    # Max absolute uv:
    uvmax = 300 # [lambda]
    logger.info('Max uv = %s' % uvmax)

    if options.CHIPS_cosmo:
        
        from astropy.cosmology import LambdaCDM
        omega_matter = 0.272
        omega_baryon = 0.046
        omega_lambda = 0.7
        hubble = 70.4
        ##Create the astropy cosmology
        cosmo = LambdaCDM(H0=hubble,
                                    Om0=omega_matter,
                                    Ode0=omega_lambda,
                                    Ob0=omega_baryon)
    else:
        # Defaults to Plank in Iris.Pspec()
        cosmo = None


    if options.sky_model != None:
        data = np.load(options.sky_model)
        Sky_mod_data = data['Sky_mod']
        N_ker_size = int(data['N_ker_size'])
        logger.info('Using user inputted sky-model %s.' % options.sky_model)

        filename = options.sky_model.split('/')[-1].split('.npz')[0]
        print(filename)
    else:
        N_ker_size = int(options.N_ker_size) # Number of sky kernel elements.

    # Number of elements in the grid.
    
    logger.info('Kernel pixel size = %s' % N_ker_size)
    N = 4*uvmax + N_ker_size # Works for all sky images.
    logger.info('Image pixel size = %s' % N)

    # List of constants.
    constants = [L,M,N,N_ker_size,uvmax,c,np.float(options.Tobs)]
    
    # Observation metadata.
    if options.zenith_pointing:
        print('Zenith beam pointing.')
        logger.info('Zenith pointed beam chosen.')
        obsid = 1080136736
        logger.info('Obsid hard set to %s'.format(obsid))
        # Loading in FITS header information from the OBSID metafits file.
        hdu = fits.getheader('{0}/models/{1}.metafits'.format(parent_path,obsid))
        chans = fits_split(hdu,'CHANNELS')[:12] # Coarse channels.
        logger.info('Coarse channels = {0}'.format((chans)))
        delays = [0.0] * 16 # Zenith pointed beam.
        logger.info('Delays = {0}'.format(delays))

        # Specifying the output file name.
        if options.out:
            name = "{0}_".format(options.out)
        else:
            # Initialising file name list:
            name = "zen_tobs{0}".format(int(options.Tobs))
        
    else:
        obsid = int(options.obsid)
        logger.info('Obsid : %s' % obsid)
        # Loading in FITS header information from the OBSID metafits file.
        hdu = fits.getheader('{0}/models/{1}.metafits'.format(parent_path,obsid))
        delays = fits_split(hdu,'DELAYS') # Delays for the beam model.
        logger.info('Delays = {0}'.format(delays))
        chans = fits_split(hdu,'CHANNELS')[:12] # Coarse channels.
        logger.info('Coarse channels = {0}'.format((chans)))
        Az_phase_centre = hdu['AZIMUTH'] # [deg]
        logger.info('Phase centre azimuth = %5.3f [deg]' % Az_phase_centre)
        Alt_phase_centre = hdu['ALTITUDE'] # [deg]
        logger.info('Phase centre altitude = %5.3f [deg]' % Alt_phase_centre)

        grid_number = hdu['GRIDNUM']
        logger.info('MWA beam sweet grid = %s' % grid_number)

        # Initialising file name list:
        # Specifying the output file name.
        if options.out:
            name = "{0}_".format(options.out)
        else:
            name = "MWA_grid{0}_tobs{1}".format(grid_number,int(options.Tobs))

    #
    ## Determining fine channels.
    #

    # Coarse channels for the observation.
    freqs = chans*1.28e+6
    waves = c/freqs

    channel = SNR_MWAbeam.channels(freqs)
    channel.calc_fine_chans(freqs)
    logger.info('Calculated the fine channels.')
    logger.info('Bandwidth = %5.2f [MHz]' % channel.bandwidth)

    #
    ## Determining the sky-model.
    #

    # Initialising the l and m vectors.
    l_vec = np.linspace(-L/2,L/2,N)
    m_vec = np.linspace(-M/2,M/2,N)
    logger.info('Calculated the direction cosine vectors.')

    ## Much cleaner way to create skymodels. Will improve with time.
    Sky_mod = Iris.Skymodel((N,N,len(channel.fine)),l_vec,m_vec)
    logger.info('Calculated the sky-model object.')

    ## There is a bug here.
    if options.plot_allsky:

        if options.zenith_pointing:
            title = 'zen'
        else:
            title = 'allsky_plot'
    else:
        pass

    #
    ## Determining the u v arrays.
    #

    # Converting the frequency of the fine channels into wavelengths.
    lam_fine_chans = c/channel.fine
    logger.info('Calculated the wavelength of the fine channels.')

    freq_cent = freqs[int(len(chans)/2)]
    logger.info('Calculated the central frequency: %5.3e [Hz]' % freq_cent)

    print('Channels : {0}'.format(chans))
    print('Bandwidth = %5.2f [MHz]' % channel.bandwidth)
    print('Central frequency = %5.2f [MHz] ' % (freq_cent/1e+6))

    ### Specifying the uv grid:
    print('Max(|uv|) = %5.2f [lambda]' % (uvmax))

    u_lam_vec = fftfreq(N,L/N)# [lambda]
    v_lam_vec = fftfreq(N,M/N)# [lambda]

    # Fixes off by one error. This otherwise causes indexing issues.
    u_lam_vec  = fftshift(u_lam_vec)
    v_lam_vec  = fftshift(v_lam_vec)
    logger.info('Calculated the (u,v) vectors.')

    # Checking that the uv_max and the min max u,v for a specific fine channel are in agreement.
    print('Min(u_arr) = %5.2f [lambda], Max(u_arr) = %5.2f [lambda]' % (np.min(u_lam_vec),np.max(u_lam_vec)))
    print('Min(v_arr) = %5.2f [lambda], Max(v_arr) = %5.2f [lambda]' % (np.min(v_lam_vec),np.max(v_lam_vec)))
    print('du = %5.2f [lambda], dv = %5.2f [lambda]' \
        % (u_lam_vec[1]-u_lam_vec[0],v_lam_vec[1]-v_lam_vec[0]))
    logger.info('du = %5.2f [lambda], dv = %5.2f [lambda]' \
        % (u_lam_vec[1]-u_lam_vec[0],v_lam_vec[1]-v_lam_vec[0]))

    # Creating the u and v plane:
    u_lam_arr,v_lam_arr = np.meshgrid(u_lam_vec,v_lam_vec)
    logger.info('Calculated the u and v grid arrays.')

    ## Determining the MWA (u,v,w) coordinates.
    # initialising the MWA array object.
    if options.gauss_array:
        MWA_array = Iris.MWA_uv(test_gauss=True)
        print('Performing test with Gaussian distributed array...')
        logger.info('Initialised Gaussian array.')
        name = name + '_gauss-arr'
    elif options.uniform_array:
        MWA_array = Iris.MWA_uv(test_uniform=True)
        print('Performing test with uniform distributed array...')
        logger.info('Initialised uniform array.')
        name = name + '_uni-arr'
    else:
        # Defualt, use the MWA Phase I array.
        MWA_array = Iris.MWA_uv()
        logger.info('Initialised MWA uv-array phase I object.')


    MWA_array.enh2xyz() # Converting array from east, north, height, to x, y, z coordinates.
    logger.info('Calculated the MWA east north height.')
    MWA_array.get_uvw() # Get the (u,v,w) coordinates, in meters.
    logger.info('Calculated the MWA array (u,v,w) coordinates.')

    if options.noise_only:
        # Option where we only calculate the noise power spectrum.
        # Can also include the 21cm signal too.
        Power = SNR_powerspec.calc_noise_powerspec(MWA_array,channel,u_lam_arr,v_lam_arr,\
            freq_cent,constants,add_21cm=options.add_21cm,wedge_cond=options.wedge_cond)
    
        print('Tobs = %5.3f [hours]' % float(options.Tobs))
        logger.info('Tobs = %5.3f [hours]' % float(options.Tobs))
        
        # Specifying the output file name.
        if options.out:
            name = "{0}_".format(options.out)
        else:
            name = 'Noise+tobs%s' % int(options.Tobs)
        
    elif options.sky_model != None:
        #
        ## Loading in user inputted model.
        #

        Sky_mod.model = Sky_mod_data
        logger.info('Sky-model object set to sky-model user inputted data.')

        # Setting flagged fine channel sky cube slices to zero.
        #Sky_mod.model[:,:,channel.flag_inds] = 0.0
        #logger.info('Flagged fine channels.')

        # Calculating the power spectrum.
        Power = SNR_powerspec.calc_powerspec(Sky_mod,MWA_array,channel,\
                u_lam_arr,v_lam_arr,freq_cent,constants,delays,add_21cm=options.add_21cm,\
                gauss_beam=options.gauss_beam,interp_cond=options.beam_interp,\
                wedge_cond=options.wedge_cond,beam_cond=options.no_beam,taper_cond=options.no_spec_taper,\
                wproj_cond=options.no_wproj)
    else:

        #
        ## Loading in the models.
        #

        if options.point_source or options.gauss_source:
            # For either a test point source or Gaussian.
            logger.info('Test source case.')
            if options.az_mod != None:
                Az = np.float(options.az_mod)
                logger.info('Test source azimuth %5.3f' % Az)
            else:
                Az = 0.0 # [deg]
                logger.info('Test source azimuth %5.3f' % Az)
            
            if options.alt_mod != None:
                Alt = np.float(options.alt_mod) # [deg]
                logger.info('Test source altitude %5.3f' % Alt)
            else:
                print('Zenith source test case...')
                Alt = 90.0 # [deg]
                logger.info('Test source altitude %5.3f' % Alt)

            if options.gauss_source:
                print('Gaussian source case...')
                logger.info('Gaussian test source case.')
                # Gaussian case.
                amaj = 1.0 #[deg]
                bmin = 1.0 #[deg]
                PA = 0 #[deg]
                logger.info('Test source major = %s [deg], minor = %s [deg], PA = %s [deg].' % (amaj,bmin,PA))
                
                name = name + "gauss_{0}_{1}".format(int(Az),int(Alt))

            else:
                print('Point source case...')
                logger.info('Point source test case.')
                name = name + "point_{0}_{1}".format(int(Az),int(Alt))

            print('Source Azimuth = %5.3f [deg]' % Az)
            print('Source Altitude = %5.3f [deg]' % Alt)
            
            # Default 1Jy flux density, with a spectral index of 0.0.
            SI = np.array([0.0]) # Spectral index.
            #S_int = np.array([1.0]) # Jy
            S_int = np.array([float(options.flux)]) # Jy

            # Amending output file name.
            name = name + "_{0}Jy".format(int(S_int[0]))

            logger.info('Test source SI = %s, flux density = %s' % (SI[0], S_int[0]))

        else:
            # Default case, read in sky-model.
            logger.info('Reading in sky-model.')
            col_names = ['Name','RA','u_RA','DEC','u_DEC','Sint','u_Sint',
                        'Maj','u_Maj','Min','u_Min','PA','u_PA','alpha','ModelID']

            logger.info('Column names = %s' % col_names)
            # Loading in supernova remnant catalogue.
            SNR_data = fits.getdata(options.model)
            t_SNR = Table(SNR_data)
            logger.info('Read in sky-model and converted to Astropy Table object.')

            colnames = t_SNR.colnames

            # Decomposing the input model into columns.
            RA = np.array(t_SNR['RA']) # [deg]
            DEC = np.array(t_SNR['DEC']) # [deg]
            logger.info('Converted Table RA and DEC to numpy array.')

            # Calculating the Azimuth and Dec.
            Alt,Az,Zen = Iris.mwa_alt_az_za(obsid, RA, DEC, degrees=True) # [deg,deg,deg]
            logger.info('Used OBSID and RA, DEC to get Alt, Az, and Zenith for each model component.')

            # Subsetting sources above the horizon.
            horizon_ind = Alt > 0.0
            logger.info('Subsetted sources that are above the horizon.')

            Alt = Alt[horizon_ind]
            Az = Az[horizon_ind]
            RA = RA[horizon_ind]
            DEC = DEC[horizon_ind]
            PA = np.array(t_SNR['PA'])[horizon_ind] # [deg]
            amaj = np.array(t_SNR['Maj'])[horizon_ind]/60 # [deg]
            bmin = np.array(t_SNR['Min'])[horizon_ind]/60 # [deg]
            S_int = np.array(t_SNR['Sint'])[horizon_ind] # [Jy]
            SI = np.array(t_SNR['alpha'])[horizon_ind]
            model_ID = np.array(t_SNR['ModelID'])[horizon_ind]

            # Specifying the output file name.
            if options.out:
                name = "{0}_".format(options.out)
            else:
                pass
                

        # Calculating the flux density for each source and each fine channel.
        S_arr = np.array([S_int*(chan/200e+6)**(-SI) for chan in channel.fine]).T
        logger.info('Calculated the flux density at each fine channel for each source.')
        logger.info('S_arr.shape = {0}'.format(S_arr.shape))

        # Calculating the power spectrum. Single and multiple cases.
        start1 = time.perf_counter()
        if options.point_source:
            # Test case for point source at zenith.
            
            Sky_mod.add_point_sources(Az, Alt, S_arr)
            logger.info('Added point source to sky-model object.')
        else:
            # Gaussian source
            if options.gauss_source:
                print('Adding zenith Gaussian to the sky-model...')
            else:
                print('Adding N = %s Gaussians to the sky-model...' % (len(RA)))
            window_size = L 
            #Sky_mod.add_Gaussian_sources(Az,Alt,amaj,bmin, PA, S_arr, window_size)
            Sky_mod.add_Gaussian_sources(Az,Alt,amaj,bmin, PA + 90, S_arr, window_size)# Rotation required.
            logger.info('Added Gaussian model(s) to sky-model object.')
        end1 = time.perf_counter()
        print('Sky-model populated time = %5.3f [s]' % (end1-start1))
        logger.info('Sky-model populated time = %5.3f [s]' % (end1-start1))

        if options.plot_allsky:
            vmax = np.nanmax(Sky_mod.model)*0.05
            print(np.nanmax(Sky_mod.model))
            Sky_mod.plot_sky_mod(title=name + '_all_sky-mod',vmax=vmax)
            logger.info('Saved plot of the sky-model.')
        else:
            pass

        # Setting flagged fine channel sky cube slices to zero.
        Sky_mod.model = Sky_mod.model # Projection effects should be accounted for.
        #Sky_mod.model[:,:,channel.flag_inds] = 0.0 # No longer flagging fine channels.
        logger.info('Flagged fine channels.')

        if options.save_sky_model:
            # If True save the sky-model, not compatible with --sky_model option.
            filename_skymod = name + '-sky_mod'
            np.savez_compressed(parent_path + 'models/' + filename_skymod, Sky_mod = Sky_mod.model, N_ker_size=N_ker_size)        
        else:
            pass


        # Calculating the power spectrum.
        Power = SNR_powerspec.calc_powerspec(Sky_mod,MWA_array,channel,\
                u_lam_arr,v_lam_arr,freq_cent,constants,delays,add_21cm=options.add_21cm,\
                gauss_beam=options.gauss_beam,interp_cond=options.beam_interp,\
                wedge_cond=options.wedge_cond,beam_cond=options.no_beam,taper_cond=options.no_spec_taper,\
                wproj_cond=options.no_wproj)
    

    if options.sky_model != None:
        # Changing name to input sky-model name.
        name = options.sky_model.split('/')[-1].split('.npz')[0]

        # Specifying the output file name.
        if options.out:
            name = "{0}_".format(options.out)
        else:
            # Adding the noise level to the name.
            name = name + '_tobs%s' % int(options.Tobs)
    else:
        pass

    if not(options.no_spec_taper):
        name = name + '_no-spec-taper'
        logger.info('Calculated power spectrum without spectral taper.')
    else:
        logger.info('Calculated power spectrum with spectral taper.')

    if not(options.no_wproj):
        name = name + '_no-wproj'
        logger.info('Calculated power spectrum without w-projection.')
    else:
        logger.info('Calculated power spectrum with w-projection.')

    if options.add_21cm:
        name = name + '+21cm'
        logger.info('Calculated power spectrum with 21cm signal.')
    else:
        logger.info('Calculated power spectrum without the 21cm signal.')
    
    if options.wedge_cond:
        name = name + '_no-wedge'
        logger.info('Calculated the power spectrum without the wedge.')
    else:
        logger.info('Calculated power spectrum with the wedge.')

    if not(options.no_beam):
        # Default option is true; we only care when this is set to FALSE.
        name = name + '_no-beam'
        logger.info('Calculated the power spectrum without a primary beam.')
    else:
        logger.info('Calculated the power spectrum without a primary beam.')

    if options.gauss_beam:
        # Default option is true; we only care when this is set to FALSE.
        name = name + '_gauss-beam'
        logger.info('Calculated the power spectrum without a Gaussian primary beam.')
    else:
        logger.info('Calculated the power spectrum without a Gaussian primary beam.')


    # Saving the 2D power spectrum.
    # Testing saving the output power arrays.
    np.savez_compressed(name, Power1D = Power.Power1D, Power2D = Power.Power2D,\
        kpar = Power.kpar, kperp = Power.kperp, k_r = Power.k_r)

    print('Saved 1D and 2D power spectra to  %s.npz' % name)
    logger.info('Saved 1D and 2D power spectra to  %s.npz' % name)
    logger.info('Saved k arrays.')

    end0 = time.perf_counter()
    print('Final time = %6.3f s' %  (end0 - start0))
    logger.info('Final time = %6.3f s' %  (end0 - start0))

    if options.plot_pspec:
        # Save power spectrum with 21cm signal inlcuded.
        Iris.Power_spec.plot_spherical(Power.k_r,Power.Power1D,title=name + '_1Dpspec',lw=3)
        logger.info('Saving 1D power spectrum.')
        Iris.Power_spec.plot_cylindrical(Power.Power2D, Power.kperp, Power.kpar, title=name + '_2Dpspec')
        logger.info('Saving 2D power spectrum.')
    else:
        pass

    # Moving files to the output directory.
    out_dir = parent_path + 'output/'
    os.system('mv *.npz %s' % out_dir)
    os.system('mv *csv %s' % out_dir)
    os.system('mv *log %s' % out_dir)
    os.system('mv *png %s' % out_dir)

else:
    pass