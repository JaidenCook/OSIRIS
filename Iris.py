#!/usr/bin/python

__author__ = "Jaiden Cook, Jack Line"
__credits__ = ["Jaiden Cook","Jack Line"]
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
warnings.simplefilter('ignore', np.RankWarning)

# Plotting stuff:
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.gridspec import GridSpec
#from matplotlib.ticker import LinearLocator, FormatStrFormatter

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

# Multiprocessing stuff:
#from joblib import Parallel, delayed
#import multiprocessing
#from tqdm import tqdm

# Scipy stuff:
import scipy
from scipy.fft import fftn,fftfreq,fftshift,ifftshift
from scipy import stats
import scipy.optimize as opt

# casa-core stuff:
#from casacore.tables import table,tablecolumn

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

def mwa_alt_az_za(obsid, ra=None, dec=None, degrees=False):
    """
    Calculate the altitude, azumith and zenith for an obsid
    Args:
    obsid : The MWA observation id (GPS time)
    ra : The right acension in HH:MM:SS
    dec : The declintation in HH:MM:SS
    degrees: If true the ra and dec is given in degrees (Default:False)
    """
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, AltAz, EarthLocation
    from astropy import units as u
   
    obstime = Time(float(obsid),format='gps')
   
    if degrees:
        sky_posn = SkyCoord(ra, dec, unit=(u.deg,u.deg))
    else:
        sky_posn = SkyCoord(ra, dec, unit=(u.hourangle,u.deg))
    earth_location = EarthLocation.of_site('Murchison Widefield Array')
    #earth_location = EarthLocation.from_geodetic(lon="116:40:14.93", lat="-26:42:11.95", height=377.8)
    altaz = sky_posn.transform_to(AltAz(obstime=obstime, location=earth_location))
    Alt = altaz.alt.deg
    Az = altaz.az.deg
    Za = 90. - Alt
    return Alt, Az, Za
       
def Gauss2D(X,Y,A,x0,y0,theta,amaj,bmin,polar=False):
    
    # By definition the semi-major axis is larger than the semi-minor axis:
    
    if amaj < bmin:
        # Swapping amaj and bmin:
        t = bmin
        bmin = amaj
        amaj = t
    else:
        pass

    # Defining the width of the Gaussians
    sigx = amaj/(2.0*np.sqrt(2.0*np.log(2.0)))
    sigy = bmin/(2.0*np.sqrt(2.0*np.log(2.0)))

    a = (np.cos(theta)**2)/(2.0*sigx**2) + (np.sin(theta)**2)/(2.0*sigy**2)
    b = -np.sin(2.0*theta)/(4.0*sigx**2) + np.sin(2.0*theta)/(4.0*sigy**2)    
    c = (np.sin(theta)**2)/(2.0*sigx**2) + (np.cos(theta)**2)/(2.0*sigy**2)
        
    if polar == False:
        # Cartesian.

        # Deriving the peak amplitude from the integrated amplitude.
        Amplitude = A/(sigx*sigy*2*np.pi)

        return Amplitude*np.exp(-(a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
    
    elif polar == True:
    
        # Arguments
        # Az,Zen,A,Az0,Zen0,theta,amaj,bmin
    
        # General 2D Gaussian function.
        # Stereographic projection.
        # 
        # https://www.aanda.org/articles/aa/full/2002/45/aah3860/node5.html
        #
        # Gaussians that exist in Spherical space are plotted onto a 2D surface.
    
        # A*exp(-(a*(x-x0)^2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)^2))
        #
        # r = 2*sin(Zen)/(1 + cos(Zen))
        #
        # x = 2*cos(Az)*sin(Zen)/(1 + cos(Zen))
        # y = 2*sin(Az)*sin(Zen)/(1 + cos(Zen))
        #
        # Zen in [0,pi]
        # Az in [0,2pi]
    

        l0 = np.sin(x0)*np.cos(y0)
        m0 = -np.sin(x0)*np.sin(y0)

        Az = X
        Zen = Y
        Az0 = x0
        Zen0 = y0
        theta_pa = theta
        
        sigx = sigx*np.sqrt((np.sin(theta_pa))**2 + (np.cos(theta_pa)*np.cos(Zen0))**2)
        sigy = sigy*np.sqrt((np.cos(theta_pa))**2 + (np.sin(theta_pa)*np.cos(Zen0))**2)

        #theta = theta + np.arctan2(l0,m0) + np.pi
        theta = theta + Az0
        
        a = (np.cos(theta)**2)/(2.0*sigx**2) + (np.sin(theta)**2)/(2.0*sigy**2)
        b = -np.sin(2.0*theta)/(4.0*sigx**2) + np.sin(2.0*theta)/(4.0*sigy**2)    
        c = (np.sin(theta)**2)/(2.0*sigx**2) + (np.cos(theta)**2)/(2.0*sigy**2)
        
        # Deriving the peak amplitude from the integrated amplitude.
        Amplitude = A/(sigx*sigy*2*np.pi)


        x_shft = np.sin(Zen)*np.cos(Az) - np.sin(Zen0)*np.cos(Az0)

        y_shft = -np.sin(Zen)*np.sin(Az) + np.sin(Zen0)*np.sin(Az0)

    
        return Amplitude*np.exp(-(a*(x_shft)**2 + 2*b*(x_shft)*(y_shft) + c*(y_shft)**2))

def Poly_func2D_nu(data_tuple,*a):
    
    (xx,yy) = data_tuple
    #
    # General form of the polynomial.
    # p(x,y) = sum^p_j=0 sum^p_i=0 b_i x^p-i y^j
    #
   
    #print "Coefficient Matrix:"
    a=np.array(a).flatten()#[0]
    #print np.shape(np.array(a).flatten())
    #print a
   
    zz = np.zeros(np.shape(xx))
    p = np.sqrt(len(a)) - 1
    p = int(p)
   
    index = 0
    for j in range(p+1):
        for i in range(p+1):
            zz = zz + a[index]*(xx**(p-i))*(yy**j)
            index += 1
            
            #print(p-i,j)
            
    return zz.ravel()

def realign_polar_xticks(ax):
    for x, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        if np.sin(x) > 0.1:
            label.set_horizontalalignment('right')
        if np.sin(x) < -0.1:
            label.set_horizontalalignment('left')

def Plot_img(Img,X_vec=None,Y_vec=None,projection='cartesian',cmap='viridis',figsize = (14,12),\
    xlab=r'$l$',ylab=r'$m$',clab='Intensity',lognorm=False,title=None,**kwargs):
    """
    Plots a 2D input image. This input image can either be in a cartesian or polar projection.
    
    Parameters
    ----------
    Img : numpy array
        2D numpy array of the image. 
    X_vec : numpy array
        1D numpy array of the x-axis. Default is 'None'.
    Y_vec : numpy array
        1D numpy array of the y-axis. Default is 'None'.
    title : string
        Name of the output image. This parameters saves the image.

    Returns
    -------
    None
    """

    if lognorm:
        norm = matplotlib.colors.LogNorm()
    else:
        norm = None

    if projection == 'cartesian':
        fig, axs = plt.subplots(1, figsize = figsize, dpi=75)

        # Creating the image objects:
        if np.any(X_vec) != None and np.any(Y_vec) != None:
            im = axs.imshow(Img,cmap=cmap,origin='upper',\
                            extent=[np.min(X_vec),np.max(X_vec),np.min(Y_vec),np.max(Y_vec)],norm=norm)
        else:
            im = axs.imshow(Img,cmap=cmap,origin='upper')
            

        # Setting the colour bars:
        cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04)
        cb.set_label(label=clab)
    
        axs.set_xlabel(xlab)
        axs.set_ylabel(ylab)
    
        im.set_clim(**kwargs)

    elif projection == "polar":
        
        fig = plt.figure(figsize = (14,12), dpi = 75)
        
        #label_size = 24
        font_size = 22
        #thetaticks = np.arange(0,360,45)
        
        ax1 = fig.add_subplot(111,projection='polar')
        pcm1 = ax1.pcolormesh(X_vec,Y_vec,Img, cmap=cmap, norm=norm)
        
        ax1.set_yticks([])
        ax1.set_theta_offset(np.pi/2.0)
        
        cb = fig.colorbar(pcm1, ax = ax1, fraction = 0.046, pad = 0.065)
    
        cb.set_label(label = 'Intensity', fontsize = font_size)
        cb.ax.tick_params(axis = 'x', labelsize = font_size - 2)
        
        realign_polar_xticks(ax1)
        
        plt.subplots_adjust(left=-0.5)
    
        pcm1.set_clim(**kwargs)
    
    if title:
        # Option for saving figure.
        plt.savefig('{0}'.format(title))
    else:
        plt.show()

def Plot_3D(X_arr,Y_arr,Z_arr,cmap='jet'):

    fontsize=24
    fig = plt.figure(figsize = (12,10), dpi=75)
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X_arr, Y_arr, Z_arr, cmap=cmap,
                       linewidth=0, antialiased=False)
    
    cb = fig.colorbar(surf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label='Intensity',fontsize=fontsize)

    ax.set_xlabel(r'$l$')
    ax.set_ylabel(r'$m$')
    
    plt.show()
    
def Plot_visibilites(Vis,N,u_vec,v_vec,cmap='viridis',lognorm=True,figsize = (14,12)):
    """
    Visibilities diagnostic plot. Plots the visibilities amplitude, the real and 
    imaginary values, and the phase for the uv-plane.
    
    Parameters
    ----------
    Img : numpy array
        2D numpy array of the image. 
    X_vec : numpy array
        1D numpy array of the x-axis. Default is 'None'.
    Y_vec : numpy array
        1D numpy array of the y-axis. Default is 'None'.
    title : string
        Name of the output image. This parameters saves the image.

    Returns
    -------
    None
    """
    if lognorm:
        norm = matplotlib.colors.LogNorm()
    else:
        norm = None

    # Creating the plots of the real, im, phase and amplitude:

    # Visibilities must be normalised before plotting.

    Vis_power = np.abs(Vis)
    
    fig, axs = plt.subplots(2,2, figsize = figsize, dpi=75)

    #Vis_power_std = np.std(Vis_power)
    #Vis_power_mean = np.mean(Vis_power)

    # Creating the image objects:
    im_Vis = axs[0,0].imshow(Vis_power,cmap=cmap,norm=norm,\
                         extent=[np.min(u_vec),np.max(u_vec),np.min(v_vec),np.max(v_vec)])#,\
                             #vmin=Vis_power_mean-5*Vis_power_std,vmax=Vis_power_mean+5*Vis_power_std)
    im_Real = axs[0,1].imshow(np.real(Vis),cmap=cmap,norm=norm,\
                          extent=[np.min(u_vec),np.max(u_vec),np.min(v_vec),np.max(v_vec)])
    im_Im = axs[1,0].imshow(np.imag(Vis),cmap=cmap,norm=norm,\
                        extent=[np.min(u_vec),np.max(u_vec),np.min(v_vec),np.max(v_vec)])
    im_Phase = axs[1,1].imshow(np.angle(Vis,deg=True),cmap=cmap,\
                           extent=[np.min(u_vec),np.max(u_vec),np.min(v_vec),np.max(v_vec)])

    # Setting the colour bars:
    cb_Vis = fig.colorbar(im_Vis, ax=axs[0,0], fraction=0.046, pad=0.04)
    cb_Vis.set_label(label='Intensity')

    cb_Real = fig.colorbar(im_Real, ax=axs[0,1], fraction=0.046, pad=0.04)
    cb_Real.set_label(label='Intensity')

    cb_Im = fig.colorbar(im_Im, ax=axs[1,0], fraction=0.046, pad=0.04)
    cb_Im.set_label(label='Intensity')

    cb_Phase = fig.colorbar(im_Phase, ax=axs[1,1], fraction=0.046, pad=0.04)
    cb_Phase.set_label(label='Phase [Degrees]')

    # Setting the axis labels:
    axs[0,0].set_xlabel(r'$u\lambda$')
    axs[0,0].set_ylabel(r'$v\lambda$')
    axs[0,0].set_title('Power')

    axs[0,1].set_xlabel(r'$u\lambda$')
    axs[0,1].set_ylabel(r'$v\lambda$')
    axs[0,1].set_title('Real Amplitude')

    axs[1,0].set_xlabel(r'$u\lambda$')
    axs[1,0].set_ylabel(r'$v\lambda$')
    axs[1,0].set_title('Imaginary Amplitude')

    axs[1,1].set_xlabel(r'$u\lambda$')
    axs[1,1].set_ylabel(r'$v\lambda$')
    axs[1,1].set_title('Phase')

    plt.show()

def Visibilities_2D(img,X,Y,N,norm=None):
    """
    This function takes an input 2D image domain array, and returns the 2D visibilities for that image.
    """
    
    # Order of Fourier operations:
    # Pad - Input array should already be padded.
    # Shift-fftshift-roll-roll necessary due to off by one error.
    # FT-fft
    # Inverse shift-ifftshift
    
    #Vis = ifftshift(fftn(np.roll(np.roll(fftshift(img),1,axis=0),1,axis=1)))
    #Vis = ifftshift(fftn(fftshift(img)))

    if norm:
        ## Noticed an off by one error in the visibilities of some low pixel size visibility images.
        Vis = np.roll(np.roll(ifftshift(fftn(np.roll(np.roll(fftshift(img),1,axis=0),1,axis=1)),norm="ortho")\
                              ,-1,axis=0),-1,axis=1)
    else:
        Vis = np.roll(np.roll(ifftshift(fftn(np.roll(np.roll(fftshift(img),1,axis=0),1,axis=1))),-1,axis=0),-1,axis=1)

    # Creating the Fourier grid:
    # N is number of sample points
    # T is sample spacing
    u_vec = fftfreq(N,X/N)
    v_vec = fftfreq(N,Y/N)

    # Creating the u and v plane:
    u_arr,v_arr = np.meshgrid(u_vec,v_vec)
    
    # Shifting the visibilities.
    u_arr = fftshift(u_arr)
    v_arr = fftshift(v_arr)[::-1,:]# For some reason these have to be flipped.

    return u_arr, v_arr, Vis

def uv_grid_kernel(N,L,M,kernel='gaussian',plot_cond=False):
    """
    This function outputs the uv degridding kernel.
    """
    
    # Defining the l and m kernel grid across the sky.
    l_ker_vec = np.linspace(-L/2,L/2,N)
    m_ker_vec = np.linspace(-M/2,M/2,N)

    l_ker_arr, m_ker_arr = np.meshgrid(l_ker_vec, m_ker_vec)

    # Initialising:
    Sky_kernel = np.zeros(np.shape(l_ker_arr))
    
    if kernel == 'gaussian':
        Sky_kernel = Gauss2D(l_ker_arr,m_ker_arr,1,0.0,0.0,0.0,0.15,0.15)
        
        # Sky kernel should sum to one. This means peak of the vis kernel will be one.
        Sky_kernel /= np.sum(Sky_kernel)
    
    if plot_cond:
        # Diagnostic plot.
        Plot_img(Sky_kernel,[-L,L],[-M,M],cmap='viridis',figsize=(7,6),clab='Weights')
    
    # Fourier transform to determine the fourier sky kernel:   
    #u_ker_lam_arr,v_ker_lam_arr,Vis_ker = Visibilities_2D(Sky_kernel,L/2,M/2,N)
    u_ker_lam_arr,v_ker_lam_arr,Vis_ker = Visibilities_2D(Sky_kernel,L,M,N)
    
    # Shifting the u, and v arrays.
    u_ker_lam_arr = (fftshift(u_ker_lam_arr)) #[lambda]
    v_ker_lam_arr = (fftshift(v_ker_lam_arr)) #[lambda]
    
    return u_ker_lam_arr, v_ker_lam_arr, Vis_ker

def Vis_degrid(u_arr,v_arr,L,M,N_ker,u,v,vis_sky,plot_cond=False,verb_cond=False,):
    """
    Visibility degridding function. Uses an input kernel, and uv point list to degrid
    visibilities.
    """
    # Initialising the new deridded visibility array:
    vis_sky_deg = np.zeros(len(u),dtype=complex)
    u_err_vec = np.zeros(len(u))
    v_err_vec = np.zeros(len(v))
    
    # might have to specify different cases. One for odd and even arrays.
    u_vec = u_arr[0,:]
    v_vec = v_arr[:,0]

    # Creating an index vector.
    ind_vec = np.arange(len(u_arr))
        
    u_pixel_size = np.abs(u_vec[0] - u_vec[1])# These should be in units of wavelengths.
    v_pixel_size = np.abs(v_vec[0] - v_vec[1])
    
    u_ker_arr, v_ker_arr, vis_ker = uv_grid_kernel(N_ker,L,M,plot_cond=plot_cond)
    
    if plot_cond == True:
        Plot_img(vis_ker.real,u_ker_arr,v_ker_arr,cmap='viridis',figsize=(7,6),xlab=r'$u(\lambda)$',\
                      ylab=r'$v(\lambda)$',clab='Weights')
    
    u_ker_pixel_size = np.abs(u_ker_arr[0,0] - u_ker_arr[0,1])# These should be in untis of wavelengths.
    v_ker_pixel_size = np.abs(v_ker_arr[0,0] - v_ker_arr[1,0])
    
    # Catch condition for degridding. Make sure pixel sizes for the kernel and the sky_vis are the same.
    if u_ker_pixel_size != u_pixel_size or v_ker_pixel_size != v_pixel_size:
        # Change this to a logger.
        print("Kernel pixel size and visibilty pixel size don't match.")
        print('du_pix = %5.2f, du_ker_pix = %5.2f' % (u_pixel_size,u_ker_pixel_size))
        print('dv_pix = %5.2f, dv_ker_pix = %5.2f' % (v_pixel_size,v_ker_pixel_size))

        return None

    # The kernel sum should equal 1.
    vis_ker = vis_ker/(np.sum(vis_ker))
    
    # Integer size of the kernel.
    ker_len = int(N_ker/2)
    
    for i in range(len(u)):
    
        # These should be the indices of the coordinates closest to the baseline. These coordinates
        # should line up with the kernel.
        temp_u_ind = ind_vec[np.isclose(u_vec,u[i],atol=u_pixel_size/2)][0]
        temp_v_ind = ind_vec[np.isclose(v_vec,v[i],atol=v_pixel_size/2)][0]

        # We also want to look at the error between the pixel position and the guess.
        u_err_vec[i] = np.abs(u[i] - u_vec[temp_u_ind])
        v_err_vec[i] = np.abs(v[i] - v_vec[temp_v_ind])

        # Might have to define a visibility subset that is larger.
        # Defining the visibility subset:
        vis_sub = vis_sky[temp_u_ind - ker_len - 1:temp_u_ind + ker_len,\
                  temp_v_ind - ker_len - 1:temp_v_ind + ker_len]
        
        #
        # Verbose output condition, for diagnostic purposes. Default condition is False.
        # Replace this with a logger, and have it raised for debugging or error purposes.
        try:
            vis_sky_deg[i] = np.sum(vis_sub*vis_ker)
        except ValueError:
            print('#{0}'.format(i))
            print('u[i] = %5.2f, v[i] = %5.2f' % (u[i],v[i]))
            print('upixel scale = %5.2f, vpixel scale = %5.2f' % (u_vec[1]-u_vec[0],v_vec[1]-v_vec[0]))
            print('min_u = %7.3f, min_v = %7.3f' % (np.min(u_vec),np.min(v_vec)))
            print('min_u_lam = %7.3f, min_v_lam = %7.3f' % (np.min(u),np.min(v)))
            print('u_diff = %5.2f, v_diff = %5.2f'% (np.min(np.abs(u_vec - u[i])),np.min(np.abs(v_vec - v[i]))))
            print('u_ind = %4i, v_ind = %4i' % (temp_u_ind,temp_v_ind))
            print('Kernel half width = %3i' % ker_len)
            print('u_ind -ker_len -1 : u_ind + ker_len = %3i : %3i' % \
                (temp_u_ind - ker_len - 1,temp_u_ind + ker_len))
            print('v_ind -ker_len -1 : v_ind + ker_len = %3i : %3i' % \
                (temp_v_ind - ker_len - 1,temp_v_ind + ker_len))

    return vis_sky_deg, u_err_vec, v_err_vec


## Interferometry functions from J.Line.

def grid(container=None,u_coords=None, v_coords=None, u_range=None, \
         v_range=None,vis=None, kernel='gaussian', kernel_params=[2.0,2.0],KERNEL_SIZE = 31):
    '''
    A simple(ish) gridder - defaults to gridding with a gaussian 
    
    Author: J.Line
    '''

    # Weight array, we will divide the entire container array by this.
    weights_arr = np.zeros(np.shape(container),dtype=complex)

    for i in np.arange(len(u_coords)):
        u,v,comp = u_coords[i],v_coords[i],vis[i]
        ##Find the difference between the gridded u coords and the current u
        ##Get the u and v indexes in the uv grdding container
        u_ind,v_ind,u_off,v_off = find_closet_uv(u=u,v=v,u_range=u_range,v_range=v_range)

        if kernel == 'gaussian':
            kernel_array = gaussian(sig_x=kernel_params[0],sig_y=kernel_params[1],\
                                    gridsize=KERNEL_SIZE,x_offset=0,y_offset=0)
            

            ker_v,ker_u = kernel_array.shape
            width_u = int((ker_u - 1) / 2)
            width_v = int((ker_v - 1) / 2)
    
            N = len(container)
            min_u_ind = u_ind - width_u
            max_u_ind = u_ind + width_u + 1
            min_v_ind = v_ind - width_v
            max_v_ind = v_ind + width_v + 1
    
            ## Jack suggests changing this.
            if max_u_ind > N-1:
                max_u_ind = N-1
                kernel_array = kernel_array[:,0:max_u_ind-min_u_ind]
    
            if max_v_ind > N-1:
                max_v_ind = N-1
                kernel_array = kernel_array[0:max_v_ind-min_v_ind,:]

            if min_u_ind < 0:
                min_u_ind = 0
                kernel_array = kernel_array[:,min_u_ind:max_u_ind]

            if min_v_ind < 0:
                min_v_ind = 0
                kernel_array = kernel_array[min_v_ind:max_v_ind,:]

            container[min_v_ind:max_v_ind, min_u_ind:max_u_ind] += comp * kernel_array
            weights_arr += kernel_array
            
        else:
            kernel_array = complex(1.0,0)
            
            container[u_ind,v_ind] += comp * kernel_array
            weights_arr[u_ind,v_ind] += kernel_array

    # Dividing the container by the sum of the weights. For natural weighting this will be the number of vis.
    container /= np.sum(weights_arr)

    return container,weights_arr

def gaussian(sig_x=None,sig_y=None,gridsize=31,x_offset=0,y_offset=0):
    '''Creates a gaussian array of a specified gridsize, with the
    the gaussian peak centred at an offset from the centre of the grid
    
    Author: J.Line
    '''

    x_cent = int(gridsize / 2.0) + x_offset
    y_cent = int(gridsize / 2.0) + y_offset

    x = np.arange(gridsize)
    y = np.arange(gridsize)
    x_mesh, y_mesh = np.meshgrid(x,y)

    x_bit = (x_mesh - x_cent)*(x_mesh - x_cent) / (2*sig_x*sig_x)
    y_bit = (y_mesh - y_cent)*(y_mesh - y_cent) / (2*sig_y*sig_y)

    amp = 1 / (2*pi*sig_x*sig_y)
    gaussian = amp*np.exp(-(x_bit + y_bit))

    return gaussian

def get_lm(ra=None,ra0=None,dec=None,dec0=None):
    '''Calculate l,m,n for a given phase centre ra0,dec0 and sky point ra,dec
    Enter angles in radians
    
    Author: J.Line
    '''

    ##RTS way of doing it
    cdec0 = np.cos(dec0)
    sdec0 = np.sin(dec0)
    cdec = np.cos(dec)
    sdec = np.sin(dec)
    cdra = np.cos(ra-ra0)
    sdra = np.sin(ra-ra0)
    l = cdec*sdra
    m = sdec*cdec0 - cdec*sdec0*cdra
    n = sdec*sdec0 + cdec*cdec0*cdra
    return l,m,n

def find_closet_uv(u=None,v=None,u_range=None,v_range=None):
    '''Finds the closet values to u,v in the ranges u_range,v_range
    Returns the index of the closest values, and the offsets from
    the closest values
    
    Author: J.Line
    '''

    u_resolution = np.abs(u_range[1] - u_range[0])
    v_resolution = np.abs(v_range[1] - v_range[0])
    
    #print(u_resolution)
    #print(v_resolution)
    
    ##Find the difference between the gridded u coords and the desired u
    u_offs = np.abs(u_range - u)

    ##Find out where in the gridded u coords the current u lives;
    ##This is a boolean array of length len(u_offs)
    u_true = u_offs < u_resolution/2.0
    
    ##Find the index so we can access the correct entry in the container
    u_ind = np.where(u_true == True)[0]

    ##Use the numpy abs because it's faster (np_abs)
    v_offs = np.abs(v_range - v)
    v_true = v_offs < v_resolution/2.0
    v_ind = np.where(v_true == True)[0]
    
    #print(u_offs,v_offs)
    
    ##If the u or v coord sits directly between two grid points,
    ##just choose the first one ##TODO choose smaller offset?
    if len(u_ind) == 0:
        u_true = u_offs <= u_resolution/2
        u_ind = np.where(u_true == True)[0]
        #print('here')
        #print(u_range.min())
    if len(v_ind) == 0:
        v_true = v_offs <= v_resolution/2
        v_ind = np.where(v_true == True)[0]
    # print(u,v)
    u_ind,v_ind = u_ind[0],v_ind[0]

    u_offs = u_range - u
    v_offs = v_range - v

    u_off = -(u_offs[u_ind] / u_resolution)
    v_off = -(v_offs[v_ind] / v_resolution)

    return u_ind,v_ind,u_off,v_off 

### Defining classes. Split this up into different module files.\

class Power_spec:
    """
    This class defines the different power spectrums. It allows for the calculation of the cylindrical and the
    angular averaged power spectrum. These are also referred to as the 2D and 1D power spectrums.
    """
    
    # Constants
    nu_21 = 1400 #[MHz]
    c = 299792458.0/1000 #[km/s]
    
    bin_width = 2.5 # [lambda]
    N_bins = 302.5/bin_width
    # Specifying the radius vector:
    k_r = np.linspace(0,302.5,int(N_bins) + 1)
    
    def __init__(self,Four_sky_cube,eta,u_arr,v_arr,nu_o,nu_21=nu_21,k_r=k_r):
        
        # Attributes will include data cube, eta, 
        
        self.data = Four_sky_cube
        self.eta = eta
        self.u_arr = u_arr # Should have units of wavelengths.
        self.v_arr = v_arr # Should have units of wavelengths.
        self.k_r = k_r
        self.z = (nu_21/nu_o) - 1
    
    
    def angular(self,Vis_power,k_r=k_r,nu_21=nu_21,c=c):
        """
        Calculate the 1D angular averaged power spectrum. This function is called for both the 1D and 2D cases.
        For the 1D case the input Vis_power data array is 3-dimensional. For the 2D cylindrical case the input
        array is 2-dimensional.
        """
        # Importing the cosmology.
        from astropy.cosmology import Planck18

        Hz = Planck18.H(self.z).value
        h = Planck18.H(0).value/100 

        # Cylindrical and non-cylindrical condition.
        shape_data = len(np.shape(Vis_power))
        
        if shape_data == 3:
            # For the non-cylindrical case, the visibilities are averaged in the time (eta) domain.
            Vis_power = np.mean(Vis_power,axis=2)
        else:
            # For the cylindrical case.
            pass
        
        # The u_arr and v_arr should be shifted. 
        r_uv = np.sqrt(self.u_arr**2 + self.v_arr**2) + 0.00001

        # Initialising Power vector and Radius vector.
        Power_spec1D = np.zeros(len(self.k_r))
        radius = np.zeros(len(self.k_r))
        for i in range(len(self.k_r)-1):
    
            radius[i] = ((self.k_r[i+1] + self.k_r[i])/2.0)
            Power_spec1D[i] = np.mean(Vis_power[np.logical_and(r_uv >= self.k_r[i], r_uv <= self.k_r[i+1])])
            
        radius = np.roll(radius,1)# Can't remember what problem this fixes.
        
        if shape_data == 2:
            # Cylindrical case.
            return Power_spec1D, radius
        else:
            # Non-cylindrical case.
            Dm = Planck18.comoving_distance(self.z).value #[Mpc]
            #self.Power_spec1D = Power_spec1D # [Jy^2 str^-2]
            self.Power_spec1D = Power_spec1D * ((c*1000)**2 / (2 * 1380.649 * nu_21**2))**2 * ((Dm/h)**2) * (h*c/Hz) # [K^2 h^-3 Mpc^3]
            self.radius = radius
            #self.kperp = self.radius * (2*np.pi/Dm) # [Mpc^-1]
            self.kperp = self.radius * (2*np.pi*h/Dm) # [h Mpc^-1]
        
    
    def cylindrical(self,k_r=k_r,c=c,nu_21=nu_21):
        """
        This code takes an input Fourier sky cube, and outputs the 2D power spectrum.
        """
        # Importing the cosmology.
        from astropy.cosmology import Planck18
    
        # Constants:
        nu_21 = 1400*1e+6 #[Hz]
        nu_o = 183*1e+6 #[Hz]
        lam_o = 1000*c/nu_o
        dnu = 30.27*1e+6 #[Hz]  # [m]full bandwidth.
        k_b = 1380.648 # [10^26 Jy m^2 K^-1] Boltzmann's constant.
        fov = 0.076 # [sr] field of view. Approximate.

        # Cosmological scaling parameter:
        h = Planck18.H(0).value/100
        E_z = Planck18.efunc(self.z)

        # Cosmological distances:
        Dm = Planck18.comoving_distance(self.z).value/h #[Mpc/h]
        DH = 3000 # [Mpc/h] Hubble distance.
        Hz = Planck18.H(self.z).value/h
    
        # Initialising th 2D power spectrum:
        Power_spec2D = np.zeros([len(self.eta),len(k_r)])
        Four_sky_cube_power = np.abs(self.data)
    
        for i in range(len(self.eta)):
        
            # Determining the k_perp power for each eta mode.
            Power_spec2D[i,:], radius = self.angular(Four_sky_cube_power[:,:,i])
        
        co_vol_factor = (Dm**2 * DH *(1+self.z)**2)/(nu_21 * E_z)

        # Converting from (Jy sr Hz)^2 to mK^2 Mpc^3 h^-3.
        conv_factor = (lam_o**4/(4*k_b**2)) * (dnu/fov) * co_vol_factor * 1e+6 #[mK^2 Mpc^3 h^-3]

        #self.Power_spec2D = Power_spec2D * ((c*1000)**2 / (2 * 1380.649 * nu_21**2))**2 * ((Dm/h)**2) * (h*c/Hz) # [K^2 h^-3 Mpc^3]
        #self.Power_spec2D = Power_spec2D #* ((c*1000)**2 / (2 * 1380.649 * nu_21**2))**2 * ((Dm/h)**2) * (h*c/Hz) # [K^2 h^-3 Mpc^3]
        self.Power_spec2D = (Power_spec2D**2)*conv_factor #* ((c*1000)**2 / (2 * 1380.649 * nu_21**2))**2 * ((Dm/h)**2) * (h*c/Hz) # [K^2 h^-3 Mpc^3]
        self.radius = radius

        print('max Power_spec2D = {0}'.format(np.max(self.Power_spec2D)))
        print('min Power_spec2D = {0}'.format(np.min(self.Power_spec2D)))


        # Calculating k_perp and k_par
        self.kperp = self.radius * (2*np.pi/Dm) # [Mpc^-1]
        #self.kperp = self.radius * (2*np.pi*h/Dm) # [h Mpc^-1]
        #self.kpar = self.eta * (2*np.pi*nu_21*Hz)/(c*(1 + self.z)**2) # [Mpc^-1]
        #self.kpar = self.eta * (2*np.pi*nu_21*Hz/h)/(c*(1 + self.z)**2) # [h Mpc^-1]
        self.kpar = self.eta * (2*np.pi*nu_21*E_z)/(DH*(1 + self.z)**2) # [h Mpc^-1]


        print(self.eta)
        print(self.kpar)
    
    def plot_angular(self,figsize = (14,12),xlim=None,ylim=None,title=None,**kwargs):
        """
        Plot the 1D angular averaged power spectrum.
        """
        
        # Initialising the figure object.
        # Need fig object, code breaks otherwise, fix this in the future.
        fig, axs = plt.subplots(1, figsize = figsize, dpi=75)
    
        plt.loglog()

        axs.plot(self.kperp,self.Power_spec1D,**kwargs)

        if xlim:
            axs.set_xlim(xlim)
        if ylim:
            axs.set_ylim(ylim)
    
        axs.set_xlabel(r'$k_\perp \,[\rm{h\,Mpc^{-1}}]$',fontsize=24)
        axs.set_ylabel(r'$\rm{P(k_\perp) \, [K^2\,h^{-3}\,Mpc^3]}$',fontsize=24)

        plt.tight_layout()
        #plt.legend(fontsize=24)

        if title:
            plt.savefig('{0}.png'.format(title))

        else:
            plt.show()

    def plot_cylindrical(self,figsize=(5.5,7),cmap='viridis',title=None,**kwargs):

        """
        Plots the 2D power spectrum.

        Parameters
        ----------

        Returns
        -------
        None
        """

        fig, axs = plt.subplots(1, figsize = figsize, dpi=75, constrained_layout=True)

    
        pspec_std = np.std(np.log10(self.Power_spec2D[self.Power_spec2D > 0]))
        pspec_min = np.min(np.log10(self.Power_spec2D[self.Power_spec2D > 0]))
        pspec_max = np.max(np.log10(self.Power_spec2D[self.Power_spec2D > 0]))

        #vmin = 10**(pspec_min + 2*pspec_std)
        #vmin = 10**(pspec_min + 1*pspec_std)
        vmin = 10#**pspec_min
        vmax = 10**pspec_max

        print('Min = %5.3e' % vmin)
        print('DC mode = %5.3e' % vmax)

        im = axs.imshow(self.Power_spec2D,cmap=cmap,origin='lower',\
                extent=[np.min(self.kperp),np.max(self.kperp),np.min(self.kpar),np.max(self.kpar)],**kwargs,\
                    norm=matplotlib.colors.LogNorm(),vmin=vmin,vmax=vmax, aspect='auto')

        # Setting the colour bars:
        cb = fig.colorbar(im, ax=axs, fraction=0.04, pad=0.002)
        #cb.set_label(label='log10 Power',fontsize=20)
        cb.set_label(label=r'$P(k_\perp,k_{||}) \, [\rm{mK^2\,h^{-3}\,Mpc^3}]$',fontsize=20)

        axs.set_xscale('log')
        axs.set_yscale('log')

        # Determined from trial and error.
        axs.set_xlim([self.kperp[3],np.max(self.kperp)])
        #axs.set_ylim([np.min(self.kpar[1:]),np.max(self.kpar)])
        axs.set_ylim([0.01,np.max(self.kpar)])

        axs.set_xlabel(r'$k_\perp \,[\rm{h\,Mpc^{-1}}]$',fontsize=20)
        axs.set_ylabel(r'$k_{||}\,[\rm{h\,Mpc^{-1}}]$',fontsize=20)

        if title:
            plt.savefig('{0}.png'.format(title))
        else:
            plt.show()


class MWA_uv:
    
    """
    Class defines the (u,v,w) coordinates for the MWA Phase I array in 
    terms of wavelengths. It take in different pointing angles. The default
    is a zenith pointing.
    """
    # Future can make it so you pass the RA and DEC of the phase centre.
    
    ## MWA latitude.
    MWA_lat = -26.703319444 # [deg] 
    ## Zenith hour angle.
    H0 = 0.0 # [deg]
    ## Array east, north, height data.
    #array_loc = np.loadtxt('antenna_locations_MWA_phase1.txt')
    array_loc = np.loadtxt('/home/jaiden/Documents/EoR/OSIRIS/antenna_locations_MWA_phase1.txt')
    
    def __init__(self,array_loc=array_loc):
        
        self.east = array_loc[:,0] # [m]
        self.north = array_loc[:,1] # [m]
        self.height = array_loc[:,2] # [m]
        
        
    def enh2xyz(self,lat=MWA_lat):
        '''Calculates local X,Y,Z using east,north,height coords,
        and the latitude of the array. Latitude must be in radians
        
        Default latitude is the MWA latitude.
        
        Author: J.Line
        '''
        lat = np.radians(lat)
        
        sl = np.sin(lat)
        cl = np.cos(lat)
        self.X = -self.north*sl + self.height*cl
        self.Y = self.east
        self.Z = self.north*cl + self.height*sl
        
        #return X,Y,Z
    def get_uvw(self,HA=H0,dec=MWA_lat):
        """
        Returns the (u,v,w) coordinates for a given pointing centre and hour angle.
        The default is a zenith pointing.
        """
        x_lengths = []
        y_lengths = []
        z_lengths = []

        # Calculating for each baseline.
        for tile1 in range(0,len(self.X)):
            for tile2 in range(tile1+1,len(self.X)):
                x_len = self.X[tile2] - self.X[tile1]
                y_len = self.Y[tile2] - self.Y[tile1]
                z_len = self.Z[tile2] - self.Z[tile1]
        
                x_lengths.append(x_len)
                y_lengths.append(y_len)
                z_lengths.append(z_len)

        # These are in metres not wavelengths.
        dx = np.array(x_lengths) # [m] 
        dy = np.array(y_lengths) # [m]
        dz = np.array(z_lengths) # [m]

        dec = np.radians(dec)
        HA = np.radians(HA)
    
        self.u_m = np.sin(HA)*dx + np.cos(HA)*dy
        self.v_m = -np.sin(dec)*np.cos(HA)*dx + np.sin(dec)*np.sin(HA)*dy + np.cos(dec)*dz
        self.w_m = np.cos(dec)*np.cos(HA)*dx - np.cos(dec)*np.sin(HA)*dy + np.sin(dec)*dz
    
    def uvw_lam(self,wavelength,uvmax):
        """
        Converts the (u,v,w) coordinates from meters to wavelengths. Additionally
        subsets for the uvmax cooridinate.
        """
    
        # Converting into wavelengths.
        u_lam = self.u_m/wavelength 
        v_lam = self.v_m/wavelength
        w_lam = self.w_m/wavelength
        
        # Determining the uv_max boolean mask
        uv_mask = (np.abs(u_lam) < uvmax)*(np.abs(v_lam) < uvmax)
        
        self.u_lam = u_lam[uv_mask]
        self.v_lam = v_lam[uv_mask]
        self.w_lam = w_lam[uv_mask]
        
        
    def plot_arr(self,uvmax,figsize=(10,10)):
        """
        Plots the MWA uv sample for a max uv cutoff. Units are in wavelengths.
        """
        plt.clf()
    
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
        ax1.plot(self.u_lam,self.v_lam,'k.',mfc='none',ms=1)
        ax1.plot(-self.u_lam,-self.v_lam,'k.',mfc='none',ms=1)
        ax1.set_xlabel(r'$u\,(\lambda)$',fontsize=24)
        ax1.set_ylabel(r'$v\,(\lambda)$',fontsize=24)
        ax1.set_xlim(-uvmax,uvmax)
        ax1.set_ylim(-uvmax,uvmax)

        plt.show()

class Skymodel:
    
    """
    Creates a sky-model class object which can be used to calculate MWA observation visibilities.

    ...

    Attributes
    ----------
    fine : numpy array
        ...


    Methods
    -------
    Gauss2D(freqs=""):
        ...
    add_Gaussian_sources(freqs=""):
        ...
    plot_sky_mod()
        
    """
    
    def __init__(self,shape,l_vec,m_vec):
        self.model = np.zeros(shape)
        
        self.l_vec = l_vec
        self.m_vec = m_vec
        
        # Creating the (l,m) plane grid:
        self.l_grid, self.m_grid = np.meshgrid(l_vec,m_vec)

        # Creating a radius array for masking purposes:
        self.r_grid = np.sqrt(self.l_grid**2 + self.m_grid**2)

        # Creating an index array, we want all pixels less than or equal to r = 1:
        self.ind_arr = self.r_grid <= 1.0

        # Here we want to create a new alt and az array that is the same size as l_arr and m_arr:
        Alt_arr = np.zeros(np.shape(self.l_grid))
        Az_arr = np.zeros(np.shape(self.l_grid))

        # Now we want to determine the Altitude and Azimuth, but only in the region where r <= 1. 
        # Outside this region isbeyond the boundary of the horizon.
        Alt_arr[self.ind_arr] = np.arccos(self.r_grid[self.ind_arr]) # Alt = arccos([l^2 + m^2]^(1/2))
        Az_arr[self.ind_arr] = np.arctan2(self.l_grid[self.ind_arr],self.m_grid[self.ind_arr]) + np.pi #arctan2() returns [-pi,pi] we want [0,2pi].
        
        # Defining the Altitude and Azimuthal grids.
        self.Alt_grid = Alt_arr
        self.Az_grid = Az_arr
        
    
    def Gauss2D(self,Az,Zen,A_tot,Az0,Zen0,theta_pa,amaj,bmin):
        """
        Generates 2D Gaussian array.

        Parameters
        ----------
        Az : numpy array, float
            2D azimuth numpy array. [deg]
        Az0 : numpy array, float
            Azimuth angle of the Gaussian centre. [deg]
        Zen : numpy array, float
            2D zenith numpy array. [deg]
        Zen0 : numpy array, float
            Zenith angle of the centre of the Gaussian. [deg]
        amaj : numpy array, float
            Gaussian major axis. [deg]
        bmin : numpy array, float
            Gaussian minor axis. [deg]
        theta_pa : numpy array, float
            Gaussian position angle. [deg]
        A_tot : numpy array, float
            Source integrated flux density.

        Returns
        -------
        2D Gaussian array.
        """        
        # General 2D Gaussian function.
        # 
        # https://www.aanda.org/articles/aa/full/2002/45/aah3860/node5.html
        #
        # Gaussians that exist in Spherical space are plotted onto a 2D surface.
    
        # A*exp(-(a*(x-x0)^2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)^2))
        #
        # r = 2*sin(Zen)/(1 + cos(Zen))
        #
        # x = 2*cos(Az)*sin(Zen)/(1 + cos(Zen))
        # y = 2*sin(Az)*sin(Zen)/(1 + cos(Zen))
        #
        # Zen in [0,pi]
        # Az in [0,2pi]
        # By definition the major axis is larger than the minor axis:
        #
        # FWHM = amaj = 2 sqrt(2 ln(2)) sigma
    
        if amaj < bmin:
            # Swapping amaj and bmin:
            t = bmin
            bmin = amaj
            amaj = t
        else:
            pass

        # Defining the width of the Gaussians
        sigx = amaj/(2.0*np.sqrt(2.0*np.log(2.0)))
        sigy = bmin/(2.0*np.sqrt(2.0*np.log(2.0)))

        sigx = sigx*np.sqrt((np.sin(theta_pa))**2 + (np.cos(theta_pa)*np.cos(Zen0))**2)
        sigy = sigy*np.sqrt((np.cos(theta_pa))**2 + (np.sin(theta_pa)*np.cos(Zen0))**2)

        # Deriving the peak amplitude from the integrated amplitude.
        Amplitude = A_tot/(sigx*sigy*2*np.pi)

        theta = theta_pa + Az0
        
        a = (np.cos(theta)**2)/(2.0*sigx**2) + (np.sin(theta)**2)/(2.0*sigy**2)
        b = -np.sin(2.0*theta)/(4.0*sigx**2) + np.sin(2.0*theta)/(4.0*sigy**2)    
        c = (np.sin(theta)**2)/(2.0*sigx**2) + (np.cos(theta)**2)/(2.0*sigy**2)

        x_shft = np.sin(Zen)*np.cos(Az) - np.sin(Zen0)*np.cos(Az0)
        y_shft = -np.sin(Zen)*np.sin(Az) + np.sin(Zen0)*np.sin(Az0)
        
        return Amplitude*np.exp(-(a*(x_shft)**2 + 2*b*(x_shft)*(y_shft) + c*(y_shft)**2))
    
    def add_Gaussian_sources(self, Az_mod, Alt_mod, Maj, Min, PA, S, window_size):
        """
        Adds 'N' number of Gaussian objects to a sky-model object. 

        Parameters
        ----------
        Az_mod : numpy array, float
            1D numpy array of the source azimuth. [deg]
        Alt_mod : numpy array, float
            1D numpy array of the source altitude. [deg]
        Maj : numpy array, float
            1D numpy array of the source major axis. [deg]
        Min : numpy array, float
            1D numpy array of the source minor axis. [deg]
        PA : numpy array, float
            1D numpy array of the source position angle. [deg]
        S : numpy array, float
            1D or 2D numpy array of Gaussian amplitudes.
        window_size : float
            Size of the Gaussian window, N*pixel_size.

        Returns
        -------
        None
        """
        # Converting the the Alt and Az into l and m coordinates:
        self.l_mod = np.cos(np.radians(Alt_mod))*np.sin(np.radians(Az_mod))# Slant Orthographic Project
        self.m_mod = -np.cos(np.radians(Alt_mod))*np.cos(np.radians(Az_mod))# Slant Orthographic Project

        if np.shape(self.l_mod):
            n_sources = len(self.l_mod)
        else:
            n_sources = 1

        for i in range(n_sources):

            # Creating temporary close l and m mask arrays:
            if np.shape(self.l_mod):
                # Multiple source case where shape(l_mod) is not None type.
                temp_l_ind = np.isclose(self.l_vec,self.l_mod[i],atol=window_size)
                temp_m_ind = np.isclose(self.m_vec,self.m_mod[i],atol=window_size)
            else:
                # Single source case.
                temp_l_ind = np.isclose(self.l_vec,self.l_mod,atol=window_size)
                temp_m_ind = np.isclose(self.m_vec,self.m_mod,atol=window_size)
    
            # Creating temporary index vectors:
            # Use the mask array to determin the index values.
            l_ind_vec = np.arange(len(self.l_vec))[temp_l_ind]
            m_ind_vec = np.arange(len(self.m_vec))[temp_m_ind]

            # Creating index arrays:
            # Use the index vectors to create arrays
            l_ind_arr, m_ind_arr = np.meshgrid(l_ind_vec, m_ind_vec)

            # Creating temporary l and m arrays:
            l_temp_arr = self.l_grid[l_ind_arr,m_ind_arr]
            m_temp_arr = self.m_grid[l_ind_arr,m_ind_arr]

            # Creating temporary Azimuth and Altitude arrays:
            ## This is the way it is described in Thompson. Section 3.1 Pg 71 Second Edition.
            Alt_temp_arr = np.arccos(np.sqrt(l_temp_arr**2 + m_temp_arr**2)) # Alt = arccos([l^2 + m^2]^(1/2))
            Az_temp_arr = np.arctan2(m_temp_arr,l_temp_arr) + np.pi  #arctan2() returns [-pi,pi] we want [0,2pi].

            # converting the major and minor axes into (l,m) coords.
            if np.shape(self.l_mod):
                # Multiple source case where shape(l_mod) is not None type.
                temp_maj = np.sin(np.radians(Maj[i]))
                temp_min = np.sin(np.radians(Min[i]))

                Gauss_temp = self.Gauss2D(Az_temp_arr, np.pi/2 - Alt_temp_arr, 1.0, 2*np.pi - np.radians(Az_mod[i]),\
                                np.pi/2 - np.radians(Alt_mod[i]),np.radians(PA[i]),\
                                temp_maj, temp_min)
            else:
                # Single source case.
                temp_maj = np.sin(np.radians(Maj))
                temp_min = np.sin(np.radians(Min))
                
                Gauss_temp = self.Gauss2D(Az_temp_arr, np.pi/2 - Alt_temp_arr, 1.0, 2*np.pi - np.radians(Az_mod),\
                                np.pi/2 - np.radians(Alt_mod),np.radians(PA),\
                                temp_maj, temp_min)
            
            self.model[l_ind_arr,m_ind_arr,:] = self.model[l_ind_arr,m_ind_arr,:] +\
                np.ones(np.shape(self.model[l_ind_arr,m_ind_arr,:]))*Gauss_temp[:,:,None]
        
            ## Set all NaNs and values below the horizon to zero:
            #self.model[self.r_arr > 1.0,:] = 0.0
            self.model[np.isnan(self.model)] = 0.0
            self.model = self.model*S[i,:]

    def add_point_sources(self, Az_mod, Alt_mod, S):
        """
        Adds 'N' number of point source objects to a sky-model object. 

        Parameters
        ----------
        Az_mod : numpy array, float
            1D numpy array of the source azimuth. [deg]
        Alt_mod : numpy array, float
            1D numpy array of the source altitude. [deg]
        S : numpy array, float
            1D or 2D numpy array of point source amplitudes.

        Returns
        -------
        None
        """
        # Converting the the Alt and Az into l and m coordinates:
        self.l_mod = np.cos(np.radians(Alt_mod))*np.sin(np.radians(Az_mod))# Slant Orthographic Project
        self.m_mod = -np.cos(np.radians(Alt_mod))*np.cos(np.radians(Az_mod))# Slant Orthographic Project

        # For the point source location.
        L = (np.max(self.l_vec) - np.min(self.l_vec))
        tol = 0.5*L/len(self.l_vec)

        if np.shape(self.l_mod):
            n_sources = len(self.l_mod)
        else:
            n_sources = 1

        for i in range(n_sources):

            # Creating temporary close l and m mask arrays:
            if np.shape(self.l_mod):
                # Multiple source case where shape(l_mod) is not None type.
                temp_l_ind = np.isclose(self.l_vec,self.l_mod[i],atol=tol)
                temp_m_ind = np.isclose(self.m_vec,self.m_mod[i],atol=tol)
            else:
                # Single source case.
                temp_l_ind = np.isclose(self.l_vec,self.l_mod,atol=tol)
                temp_m_ind = np.isclose(self.m_vec,self.m_mod,atol=tol)
    
            # Creating temporary index vectors:
            # Use the mask array to determin the index values.
            l_ind_vec = np.arange(len(self.l_vec))[temp_l_ind]
            m_ind_vec = np.arange(len(self.m_vec))[temp_m_ind]

            # Setting point source value:
            self.model[l_ind_vec,m_ind_vec,:] = 1.0
        
            ## Set all NaNs and values below the horizon to zero:
            self.model[np.isnan(self.model)] = 0.0
            self.model = self.model*S[i,:]

    def plot_sky_mod(self,window=None,index=None,figsize=(14,14),xlab=r'$l$',ylab=r'$m$',cmap='viridis',\
        clab=None,title=None,**kwargs):
        """
        This function plots a subset of the sky-model. Particularly for a single source.
        The main purpose of the functions in this pipeline is to plot the visibilities 
        for a single source. Additionally there is an all-sky plotting option.
        """

        fig, axs = plt.subplots(1, figsize = figsize, dpi=75)

        if index:
            # Case for a single source, when there is more than one model source.
            l_mod = self.l_mod[index]
            m_mod = self.m_mod[index]
        else:
            # Case for a single source, when there is more than one model source.
            l_mod = self.l_mod
            m_mod = self.m_mod

        if window:
            # Case for a single source image.
            # Specifying the temporary l and m indices based on window size.
            temp_l_ind = np.isclose(self.l_vec,l_mod,atol=window)
            temp_m_ind = np.isclose(self.m_vec,m_mod,atol=window)

            # Creating temporary index vectors:
            # Use the mask array to determin the index values.
            l_ind_vec = np.arange(len(self.l_vec))[temp_l_ind]
            m_ind_vec = np.arange(len(self.m_vec))[temp_m_ind]

            # Creating index arrays:
            # Use the index vectors to create arrays
            l_ind_arr, m_ind_arr = np.meshgrid(l_ind_vec, m_ind_vec)

            # Creating temporary l and m arrays:
            l_temp_arr = self.l_grid[l_ind_arr,m_ind_arr]
            m_temp_arr = self.m_grid[l_ind_arr,m_ind_arr]

            im = axs.imshow(self.model[l_ind_arr,m_ind_arr,100],cmap=cmap,origin='upper',\
                extent=[np.min(l_temp_arr),np.max(l_temp_arr),np.min(m_temp_arr),np.max(m_temp_arr)])
        else:
            # Case for the whole sky.
            im = axs.imshow(self.model[:,:,100],cmap=cmap,origin='lower',\
                extent=[np.min(self.l_grid),np.max(self.l_grid),np.min(self.m_grid),np.max(self.m_grid)])

        if clab:
            # Find a better way to do this.
            clab = clab
        else:
            # Default colour bar label.
            clab = r'$I_{\rm{app}}\,[\rm{Jy/Str}]$'

        # Setting the colour bars:
        cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04)
        cb.set_label(label=clab,fontsize=24)

        axs.set_xlabel(xlab,fontsize=24)
        axs.set_ylabel(ylab,fontsize=24)

        im.set_clim(**kwargs)

        if title:
            plt.savefig('{0}.png'.format(title))
        else:
            plt.show()