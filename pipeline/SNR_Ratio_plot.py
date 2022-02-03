#!/usr/bin/python

__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "0.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"

# Generic stuff:
#%matplotlib notebook
import os,sys
from datetime import datetime
from math import pi
import warnings
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


sys.path.append(os.path.abspath("/home/jaiden/Documents/EoR/OSIRIS"))
import Osiris
from Iris_degrid import *
from Iris_grid import *
import SNR_MWAbeam
#import SNR_21cmps



if __name__ == '__main__':
    # Defining the parser:
    usage="Usage: %prog [options]\n"
    parser = OptionParser(usage=usage)
    parser.add_option('--2Dpspec1',dest="pspec2D_1",default=None,help="Input 2D power spectrum, this will be the numerator.\n")
    parser.add_option('--2Dpspec2',dest="pspec2D_2",default=None,help="Input 2D power spectrum, this will be the deonominator. Requires option --2Dpspec1.\n")
    parser.add_option('--max',dest="vmax",default=None,help="Input 2D power spectrum max. Requires option --2Dpspec1.\n")
    parser.add_option('--min',dest="vmin",default=None,help="Input 2D power spectrum min. Requires option --2Dpspec1.\n")
    parser.add_option('--xlim',dest="xlim",default=None,type='str',help="Input k_perp lim. Requires option --2Dpspec1.\n")
    parser.add_option('--ylim',dest="ylim",default=None,type='str',help="Input k_|| lim. Requires option --2Dpspec1.\n")
    parser.add_option('--calc_window',dest="calc_window",default=False,action='store_true',help="Calculate Average EoR window power. Use with xlim and ylim. Requires option --2Dpspec1.\n")
    parser.add_option('--title',dest="title",default=None,help="Input 2D power spectrum min. Requires option --2Dpspec1.\n")
    parser.add_option('--1Dpspec',dest="pspec1D",default=None,help="Input 1D power spectrum.\n")
    parser.add_option('--outname',dest="outname",default=None,help="Output image name.\n")
    parser.add_option('--tobs',dest="tobs",default=None,help="Observations time.\n")
    parser.add_option('--pathname',dest="pathname",default=None,help="Input file path name.\n")
    parser.add_option('--add_1D_noise',dest="add_1D_noise",default=False,action='store_true',help="If true then add 1D noise to 1D pspec plot. Must have noise file in the same directory.\n")

    (options, args) = parser.parse_args()

    if options.pathname:
        parent_path = options.pathname
    else:
        #parent_path = '/home/jaiden/Documents/EoR/SNR-Pipeline/output/no_flagging/'
        parent_path = '/home/jaiden/Documents/EoR/SNR-Pipeline/output/'

    if options.pspec2D_1 or options.pspec2D_2:
        # Options for taking the ratio of two 2D power spectra.

        if options.pspec2D_1 and options.pspec2D_2:
            # Using a different file format.
            data1 = np.load(parent_path + options.pspec2D_1)
            data2 = np.load(parent_path + options.pspec2D_2)
        
            Pspec2D_1 = data1['Power2D']
            kpar = data1['kpar']
            kperp = data1['kperp']

            # Loading in the second Pspec.
            Pspec2D_2 = data2['Power2D']

            Power_arr = Pspec2D_1/Pspec2D_2

            label = r'$P_1(k_\perp,k_{||})/P_2(k_\perp,k_{||}) $'

            print('Min 2D Ratio = %5.3f' % np.nanmin(Power_arr))
            print('Median 2D Ratio = %5.3f' % np.nanmedian(Power_arr))
            print('Max 2D Ratio = %5.3f' % np.nanmax(Power_arr))
        
        elif options.pspec2D_1 and not(options.pspec2D_2):
            # Plotting a single 2D power spectra.
        
            # Loading in the data.
            data1 = np.load(parent_path + options.pspec2D_1)

            kpar = data1['kpar']
            kperp = data1['kperp']

            Power_arr = data1['Power2D']

            label = r'$P_(k_\perp,k_{||})$'


        if options.vmax != None:
            vmax = float(options.vmax)
        else:
            vmax = options.vmax

        if options.title != None:
            title = str(options.title)
        else:
            title = options.title
        
        if options.vmin != None:
            vmin = float(options.vmin)
        else:
            vmin = options.vmin

        if options.xlim:
            xlim = eval(options.xlim)
        else:
            xlim = None

        if options.ylim:
            ylim = eval(options.ylim)
        else:
            ylim = None

        if options.calc_window and xlim!= None:

            par_window_ind_vec = np.arange(len(kpar))[(kpar >= ylim[0])*(kpar <= ylim[1])]
            perp_window_ind_vec = np.arange(len(kperp))[(kperp >= xlim[0])*(kperp <= xlim[1])]

            # Creating 2D index arrays.
            xx_ind,yy_ind = np.meshgrid(par_window_ind_vec,perp_window_ind_vec)

            # Creating subset
            Power_arr_window = Power_arr[xx_ind,yy_ind]
            Power_arr_window = Power_arr_window[np.isnan(Power_arr_window) == False]

            N_bins = len(Power_arr_window)

            print('N = %3i' % N_bins)

            # Plot histogram for analysis purposes.
            #plt.hist(Power_arr_window.flatten())

            y_hist,x_hist_edges = np.histogram(Power_arr_window,bins=int(np.sqrt(N_bins)),density=True)

            x_hist = np.array([(x_hist_edges[i+1] + x_hist_edges[i])/2 for i in range(len(x_hist_edges)-1)])

            print('Mean window power = %5.3f +- %5.3f' % (np.nanmean(Power_arr_window),np.sqrt(0.429)*np.nanstd(Power_arr_window)))
            print('Median window power = %5.3f +- %5.3f' % (np.nanmedian(Power_arr_window),np.sqrt(0.429)*np.nanstd(Power_arr_window)))

            # Testing that the windowing works.
            #Power_arr[xx_ind,yy_ind] = 100
        else:
            pass

        if options.xlim != None: 
            if xlim[0] < np.min(kperp):
                # Fixes plotting issue.
                xlim = [0.008,xlim[1]]
        else:
            pass

        if options.outname:
            #Osiris.Power_spec.plot_cylindrical(ratio_2Dpspec,kperp,kpar,lognorm=True,clab=r'$\rm{Ratio}$',title=options.outname)
            Osiris.Power_spec.plot_cylindrical(Power_arr,kperp,kpar,lognorm=True,clab=label,name=options.outname,\
                vmax=vmax,vmin=vmin,figsize=(8.5*(4./5.),10.5*(4./5.)),xlim=xlim,ylim=ylim,title=title)

            # Move all plots to the output folder.
            os.system('mv *png %s' % parent_path)
        else:
            #Osiris.Power_spec.plot_cylindrical(ratio_2Dpspec,kperp,kpar,lognorm=True,clab=r'$\rm{Ratio}$')
            #Osiris.Power_spec.plot_cylindrical(ratio_2Dpspec,kperp,kpar,lognorm=True,clab=label)
            Osiris.Power_spec.plot_cylindrical(Power_arr,kperp,kpar,lognorm=True,clab=label,\
                vmax=vmax,vmin=vmin,figsize=(8.5*(4./5.),10.5*(4./5.)),xlim=xlim,ylim=ylim,title=title)

    

    elif options.pspec1D:

        # Constants:
        c = 299792458.0/1000 #[km/s]
        chans = np.array([131,132,133,134,135,136,137,138,139,140,141,142])
        freqs = chans*1.28e+6

        # Calculating the channels:
        channel = SNR_MWAbeam.channels(freqs)
        channel.calc_fine_chans(freqs)

        # Channel bandwidth:
        dnu = channel.bandwidth # MHz
        dnu_f = channel.fine_chan_width/1e+6 #MHz

        # Calculating the central frequency.
        chan_cent = 0.5*(chans[-1] + chans[0])*1.28
        freq_cent = chan_cent*1e+6

        pspec1D_path = '/home/jaiden/Documents/EoR/21cmpspec/model/'
        file_seed = pspec1D_path + 'ps_no_halos_z'

        # Number of characters.
        N_char = len(file_seed)

        # Channel bandwidth:
        dnu = channel.bandwidth # MHz
        dnu_f = channel.fine_chan_width/1e+6 #MHz
        f21 = (1000*c)/(0.21) #[Hz]
        z = (f21)/freq_cent - 1

        filenames = os.popen('find {0}*'.format(file_seed)).read().split('\n')[:-1]

        redshift_vec = np.array([float(file[N_char:(N_char + 6)]) for file in filenames])
        pspec_names = [file[len(pspec1D_path):] for file in filenames]

        file_index = np.argmin(np.abs(redshift_vec-z))
        file_cent = pspec_names[file_index]

        # Loading in the array:
        Data_array = np.loadtxt(pspec1D_path + file_cent, usecols=(0,1,2))

        k_r = Data_array.T[0,:]
        norm_factor = ((2*np.pi**2)/k_r**3)
        Power1D_21cm = Data_array.T[1,:]*norm_factor

        #
        ## Loading in 1D power spectrum.
        #

        # Using the new data format.
        data = np.load(parent_path + options.pspec1D)

        Power1D_k_r = data['k_r']
        Power1D_sim = data['Power1D']

        #
        ## Plotting
        #

        fig, axs = plt.subplots(1, figsize = (8,6))

        #Osiris.Power_spec.plot_spherical(k_r,Power1D_21cm,figaxs=(fig,axs),lw=3,label='21cm signal')
        Osiris.Power_spec.plot_spherical(Power1D_k_r,Power1D_sim,figaxs=(fig,axs),ls='--',lw=3,label='Simulation')

        if options.add_1D_noise:

            tail = '_21cm_1Dpspec.csv'
            head = 'Noise+tobs'

            tobs_split_str = options.pspec1D.split('tobs')[1]

            #print(tobs_split_str)
            #print(len(tobs_split_str))

            if options.tobs != None:
                tobs = int(options.tobs)
            else:
                if len(tobs_split_str.split('+')) >1:
                    tobs = int(tobs_split_str.split('+')[0])
                elif len(tobs_split_str.split('_')) >1:
                    tobs = int(tobs_split_str.split('_')[0])
                else:
                    pass
                    #t0bs = int(options.add)

            filename = head + str(int(tobs)) + '.npz'

            # Using the new data format.
            data_noise = np.load(parent_path + filename)

            Power1D_k_r = data_noise['k_r']
            Power1D_noise = data_noise['Power1D']

            Osiris.Power_spec.plot_spherical(Power1D_k_r,Power1D_noise,figaxs=(fig,axs),ls='--',lw=3,label='Noise t = %s hrs' % tobs)
        else:
            pass


        plt.legend(fontsize=18)
        plt.tight_layout()

        if options.outname:
            plt.savefig('%s.png' % options.outname)
            # Move all plots to the output folder.
            os.system('mv *png %s' % parent_path)
        else:
            plt.show()

    else:

        pass

else:
    pass

