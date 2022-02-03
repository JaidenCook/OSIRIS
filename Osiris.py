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

from tqdm import tqdm

# Array stuff:
import numpy as np
warnings.simplefilter('ignore', np.RankWarning)

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
from scipy.fft import ifftn,fftn,fftfreq,fftshift,ifftshift
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
    

        #l0 = np.sin(x0)*np.cos(y0)
        #m0 = -np.sin(x0)*np.sin(y0)

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

def Plot_img(Img,X_vec=None,Y_vec=None,projection='cartesian',cmap='cividis',figsize = (14,12),\
    xlab=r'$l$',ylab=r'$m$',clab='Intensity',lognorm=False,title=None,\
    clim=None,vmin=None,vmax=None,contours=None,**kwargs):
    """
    Plots a 2D input image. This input image can either be in a cartesian or polar projection.
    
    Parameters
    ----------
    Img : numpy array
        2D numpy array of the image. 
    X_vec : numpy array
        1D numpy array of the x-axis/Azimuth-axis. Default is 'None'.
    Y_vec : numpy array
        1D numpy array of the y-axis/radius-axis. Default is 'None'.
    title : string
        Name of the output image. This parameters saves the image.
    contours : int
        Contour level. Default is None.

    Returns
    -------
    None
    """

    def realign_polar_xticks(ax):
        for x, label in zip(ax.get_xticks(), ax.get_xticklabels()):
            if np.sin(x) > 0.1:
                label.set_horizontalalignment('right')
            if np.sin(x) < -0.1:
                label.set_horizontalalignment('left')

    if lognorm:
        norm = matplotlib.colors.LogNorm()
    else:
        norm = None

    if np.any(vmax):
        vmax=vmax
    
    if np.any(vmin):
        vmin=vmin

    cmap = matplotlib.cm.viridis
    cmap.set_bad('lightgray',1.)

    if projection == 'cartesian':
        fig, axs = plt.subplots(1, figsize = figsize, dpi=75)

        # Creating the image objects:
        if np.any(X_vec) != None and np.any(Y_vec) != None:

            im = axs.imshow(Img,cmap=cmap,origin='lower',\
                    extent=[np.min(X_vec),np.max(X_vec),np.min(Y_vec),np.max(Y_vec)],\
                    norm=norm,vmin=vmin,vmax=vmax,**kwargs)
        else:
            im = axs.imshow(Img,cmap=cmap,origin='lower',norm=norm,\
                vmin=vmin,vmax=vmax,**kwargs)

        if contours:
            axs.clabel(contours, inline=True, fontsize=8)
        else:
            pass

        # Setting the colour bars:
        if np.any(vmax):
            cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, format='%.1e',extend='max')
        else:
            cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, format='%.1e')

        cb.set_label(label=clab,fontsize=20)
        cb.ax.tick_params(labelsize=20)

        axs.set_xlabel(xlab,fontsize=20)
        axs.set_ylabel(ylab,fontsize=20)
    
        axs.tick_params(axis='both', labelsize=20)

        im.set_clim(clim)

    elif projection == "polar":
        
        fig = plt.figure(figsize = figsize, dpi = 75)

        #label_size = 24
        font_size = 22
        #thetaticks = np.arange(0,360,45)

        ax1 = fig.add_subplot(111,projection='polar')
        pcm1 = ax1.pcolormesh(X_vec,Y_vec,Img,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
        
        ax1.set_yticks([])
        ax1.set_theta_offset(np.pi/2.0)
        
        cb = fig.colorbar(pcm1, ax = ax1, fraction = 0.046, pad = 0.065)
    
        cb.set_label(label = 'Intensity', fontsize = font_size)
        cb.ax.tick_params(axis = 'x', labelsize = font_size - 2)
        
        ax1.set_ylim([0,1.])

        realign_polar_xticks(ax1)
        
        plt.subplots_adjust(left=-0.5)
    
        pcm1.set_clim(clim)
    
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

def Visibilities_2D(img,X=None,Y=None,N=None,norm=None):
    '''
    This function takes an input 2D image domain array, and returns the 2D visibilities for that image.

        Parameters
        ----------
        img : numpy array, float
            2D image array.
        X : numpy array, float
            Image array size, default is 'None'.
        Y : numpy array, float
            Image array size, default is 'None'.
        N : numpy array, float
            Image array pixel size, default is 'None'.
        norm : string
            Norm condition, accepts 'forward', 'backward', and 'ortho', default is 'backward'.

        Returns
        -------
        2D complex visibility array.

    '''

    # Order of Fourier operations:
    # Pad - Input array should already be padded.
    # Shift-fftshift-roll-roll necessary due to off by one error.
    # FT-ifft
    # Inverse shift-ifftshift
    
    ## Default original case:
    #print('3D case')
    Vis = np.roll(np.roll(ifftshift(fftn(np.roll(np.roll(fftshift(img),1,axis=0),1,axis=1),norm=norm)),-1,axis=0),-1,axis=1)
    #Vis = np.roll(np.roll(ifftshift(ifftn(np.roll(np.roll(fftshift(img),1,axis=0),1,axis=1),norm=norm)),-1,axis=0),-1,axis=1)

    if X and Y and N:
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
    else:
        # Default case:
        return Vis

def Img_slice(Vis_slice,Nb):
    '''
    Performs the inverse FFT on a 2D visibility grid.

        Parameters
        ----------
        vis_slice : numpy array, complex128
            2D visiblity array. SHould have odd sized dimensions.
        Nb : int
            Number of baselines.
       
        Returns
        -------
        2D image array.

    '''
    Vis_slice = Vis_slice/Nb

    I_out = (np.roll(fftshift(ifftn(np.roll(np.roll(ifftshift(Vis_slice),
                        1,axis=0),1,axis=1),norm='forward')),-1,axis=0))
    
    return I_out

def gaussian_kernel(u_arr,v_arr,sig_u,sig_v,u_cent,v_cent):
    '''
    Generate A generic 2D Gassian kernel. For gridding and weighting purposes.

        Parameters
        ----------
        u_arr : numpy array, float
            2D Visibilities u array.
        v_arr : numpy array, float
            2D Visibilities v array.
        sig_u : numpy array, float
            Kernel size in u.
        sig_v : numpy array, float
            Kernel size in v.
        u_cent : numpy array, float
            Visibility u coordinate centre.
        v_cent : numpy array, float
            Visibility v coordinate centre.

        Returns
        -------
        2D Gaussian weights array.

    '''

    u_bit = (u_arr - u_cent)/sig_u
    v_bit = (v_arr - v_cent)/sig_v

    amp = 1/(2*np.pi*sig_u*sig_v)
    gaussian = amp*np.exp(-0.5*(u_bit**2 + v_bit**2))

    # Normalising so the sum of the gaussian is equal to 1.
    gaussian = gaussian/np.sum(gaussian)

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

def find_closest_xy(x,y,x_vec,y_vec,off_cond=False):
    '''
    Finds the indices for the (x,y) point associated to a x, y grid.

        Parameters
        ----------
        x : numpy array, float
            x value.
        y : numpy array, float
            y value.
        x_vec : numpy array, float
            Regular 1D uxgrid.
        y_vec : numpy array, float
            Regular 1D y grid.

        Returns
        -------
        Returns closest (x,y) indices.

    Author : J. Line
    Modified by J. Cook
    '''
    x_resolution = np.abs(x_vec[1] - x_vec[0])
    y_resolution = np.abs(y_vec[1] - y_vec[0])
    
    ##Find the difference between the gridded u coords and the desired u
    x_offs = np.abs(x_vec - x)

    ##Find out where in the gridded u coords the current u lives;
    ##This is a boolean array of length len(u_offs)
    x_true = x_offs < x_resolution/2.0
    
    ##Find the index so we can access the correct entry in the container
    x_ind = np.where(x_true == True)[0]

    ##Use the numpy abs because it's faster (np_abs)
    y_offs = np.abs(y_vec - y)
    y_true = y_offs < y_resolution/2.0
    y_ind = np.where(y_true == True)[0]

    ##If the u or v coord sits directly between two grid points,
    ##just choose the first one ##TODO choose smaller offset?
    if len(x_ind) == 0:
        x_true = x_offs <= x_resolution/2
        x_ind = np.where(x_true == True)[0]
        #print('here')
        #print(u_range.min())
    if len(y_ind) == 0:
        y_true = y_offs <= y_resolution/2
        y_ind = np.where(y_true == True)[0]
    # print(u,v)
    x_ind, y_ind = x_ind[0], y_ind[0]
    #x_ind, y_ind = x_ind, y_ind

    if off_cond:
        # Offset condition, if true return the offsets alongside the indices.
        x_offs = x_vec - x
        y_offs = y_vec - y

        # Offsets need to be multiplied by negative one to phase shift in the correct direction.
        x_offs = -1*x_offs[x_ind]
        y_offs = -1*y_offs[y_ind]

        return x_ind, y_ind, x_offs, y_offs
    else:
        # Default condition don't return the offsets.
        return x_ind,y_ind

### Defining classes. Split this up into different module files.\

class Power_spec:
    """
    This class defines the different power spectrums. It allows for the calculation of the cylindrical and the
    angular averaged power spectrum. These are also referred to as the 2D and 1D power spectrums.

    ...

    Attributes
    ----------
    fine : numpy array
        ...


    Methods
    -------
    Power2Tb(freqs=""):
        ...
    uv_to_kxky(freqs=""):
        ...
    eta_to_kz(freqs=""):
        ...
    wedge_factor()
        ...
    Spherical()
        ...
    Cylindrical()
        ...
    plot_spherical()
        ...
    plot_cylindrical()
    """

    # Constants
    c = 299792458.0/1000 #[km/s]
    nu_21 = (1000*c)/(0.21) #[Hz]
    kb = 1380.649 # [Jy m^2 Hz K^-1] Boltzmann's constant.
    
    def __init__(self,Four_sky_cube,eta,u_arr,v_arr,nu_o,dnu,dnu_f,weights_cube=None,nu_21=nu_21,cosmo=None):
        
        # Attributes will include data cube, eta, 
        # For determining the 2D bins.
        #self.uvmax = np.max(u_arr)
        self.uvmax = 300

        #self.data = Four_sky_cube
        self.power_cube = np.conjugate(Four_sky_cube)*Four_sky_cube # [Jy^2 Hz^2]

        if np.any(weights_cube):
            # Case for user inputted weigth cube.
            self.weights_cube = weights_cube

        else:
            print('Natural weighting case.')
            # Default weighting scheme is natural. Default set to not break older code.
            self.weights_cube = np.zeros(np.shape(Four_sky_cube))
            
            # Only cells with values are assigned weights.
            self.weights_cube[self.power_cube > 0.0] = 1.0

            #print(np.sum(self.weights_cube))
        
        self.eta = eta # [Hz^-1]
        self.u_arr = u_arr # Should have units of wavelengths.
        self.v_arr = v_arr # Should have units of wavelengths.
        self.nu_o = nu_o # [Hz]
        self.z = (nu_21/self.nu_o) - 1
        self.dnu = dnu # Bandwidth in [MHz].
        self.dnu_f = dnu_f # Fine channel width in [MHz].
        print('Redshift = %5.2f' % self.z)

        if cosmo != None:
            # User inputted cosmology.
            print('Using non-standard cosmology.')
            self.cosmo = cosmo
        else:
            # Default is the Plank18 cosmology.
            from astropy.cosmology import Planck18

            self.cosmo = Planck18


        # Save memory.
        del Four_sky_cube
    
    @staticmethod
    def Power2Tb(dnu,dnu_f,nu_o,z,cosmo,verbose=True):
        """
        Calculate the conversion factor from Jy^2 Hz^2 to mK^2 Mpc^3 h^-3.

            Parameters
            ----------
            dnu : float
                Bandwidth [Hz].
            dnu_f : float
                Fine channel width [Hz].
            nu_o : float
                Observing frequency at the centre of the band [Hz].
            z : float
                Redshift of the observing frequency.
            cosmo : astropy object
                Astropy Cosmology object, default used is Plank2018.
            
            Returns
            -------
            conv_factor
        """
        from scipy import signal

        # Constants
        c = 299792458.0/1000 #[km/s]
        nu_21 = c*1000/(0.21) #[Hz]
        kb = 1380.649 # [Jy m^2 Hz K^-1] Boltzmann's constant.

        # Constants.
        lam_21 = 1000*c/nu_21 #[m]
        #lam_o = 1000*c/self.nu_o #[m]
        lam_o = 1000*c/nu_o #[m]
        fov = 0.076 # [sr] field of view. Approximate.
        N_chans = dnu/dnu_f

        if verbose:
            print('Observed wavelength = %5.3f [m]' % lam_o)
            print('Fine channel width = %5.3e' % dnu_f)
        else:
            pass
        
        # Calculating the volume correction factor:
        window = signal.blackmanharris(int(dnu/dnu_f))
        Ceff = np.sum(window)/(dnu/dnu_f)

        if verbose:
            print('Volume correction factor = %5.3f' % (Ceff))
        else:
            pass

        # Cosmological scaling parameter:
        h = cosmo.H(0).value/100
        #E_z = cosmo.efunc(self.z)
        E_z = cosmo.efunc(z)

        # Cosmological distances:
        Dm = cosmo.comoving_distance(z).value*h #[Mpc/h]
        DH = 3000 # [Mpc/h] Hubble distance.

        # Volume term.
        co_vol = (1/Ceff)*(Dm**2 * DH *(1 + z)**2)/(nu_21 * E_z) # [sr^-1 Hz^-1 Mpc^3 h^-3]

        if verbose:
            print('Volume term = %5.3f [sr^-1 Hz^-1 Mpc^3 h^-3]' % co_vol)
        else:
            pass

        # Converting a 1 Jy^2 source to mK^2 Mpc^3 h^-3.
        conv_factor = (N_chans**2) * (lam_o**4/(4*kb**2)) * (1/(fov*dnu)) * co_vol * 1e+6 # [mK^2 Mpc^3 h^-3]

        if verbose:
            print('Conversion factor = %5.3f [mK^2 Hz^-2 Mpc^3 h^-3]' % conv_factor)
            print('Conversion factor = %5.3f [mK^2 Hz^-2 Mpc^3]' % (conv_factor*h**3))
        else:
            pass
        
        return conv_factor

    @staticmethod
    def uv_to_kxky(u,z,cosmo):
        """
        Convert u or v into k_x or k_y, k_z as per Morales et al. (2004).
        Uses the Plank 2018 cosmology as default. 

        Can convert r = sqrt(u^2 + v^2) to k_perp. Same formula.
                
        Parameters
            ----------
            u_arr : numpy array, float
                NDarray of u or v values. Should be in wavelengths.
            z : float
                Redshift at the central frequency of the band.
            
            Returns
            -------
            k_vec : numpy array, float
                NDarray of k-mode values. Should be in units of h*Mpc^-1. 
        """

        # Cosmological scaling parameter:
        h = cosmo.H(0).value/100 # Hubble parameter.
    
        # Cosmological distances:
        Dm = cosmo.comoving_distance(z).value*h #[Mpc/h] Transverse co-moving distance.

        # Converting u to k
        k_vec = u * (2*np.pi/Dm) # [Mpc^-1 h]

        return k_vec

    @staticmethod
    def eta_to_kz(eta,z,cosmo):
        """
        Convert eta into k_z as per Morales et al. (2004).
        Uses the Plank 2018 cosmology as default.
                
        Parameters
            ----------
            eta : numpy array, float
                1Darray of eta values. 
            z : float
                Redshift at the central frequency of the band.
            
            Returns
            -------
            k_z : numpy array, float
                1Darray of kz values. Should be in units of h*Mpc^-1.
        """

        # Constant:
        c = 299792458.0/1000 #[km/s]
        nu_21 = (1000*c)/(0.21) #[Hz]

        # Cosmological scaling parameter:
        E_z = cosmo.efunc(z) ## Scaling function, see (Hogg 2000)

        # Cosmological distances:
        DH = 3000 # [Mpc/h] Hubble distance.

        # k_||
        k_z = eta * (2*np.pi*nu_21*E_z)/(DH*(1 + z)**2) # [Mpc^-1 h]

        return k_z

    @staticmethod
    def wedge_factor(z,cosmo):
        """
        Nicholes horizon cosmology cut.
                
        Parameters
            ----------
            z : float
                Redshift.
            cosmo : Astropy Object
                Astropy cosmology object, default is None. If None use Plank18 cosmology.
            
            Returns
            -------
            wedge_factor : float
                k|| > wedge_factor * k_perp cut.
        """

        # Cosmological scaling parameter:
        h = cosmo.H(0).value/100 # Hubble parameter.
        E_z = cosmo.efunc(z) ## Scaling function, see (Hogg 2000)

        # Cosmological distances:
        Dm = cosmo.comoving_distance(z).value*h #[Mpc/h] Transverse co-moving distance.
        DH = 3000 # [Mpc/h] Hubble distance.

        wedge_factor = Dm*E_z/(DH*(1 + z)) 

        return wedge_factor

    def Spherical(self,kb=kb,nu_21=nu_21,c=c,wedge_cond=False):
        """
        Calculates the 1D spherical power spectra using the input Power object.
                
        Parameters
            ----------
            self : object
                Power object contains u and v arrays, as well as the observation redshift.
            kb : float
                Boltzman's constant.
            nu_21 : float
                21cm frequency in Hz.
            kb : float
                Speed of light km/s.
            
            Returns
            -------
        """

        fov = 0.076
        vol_fac = 0.35501

        # Defining the kx, ky, and kz values from u,v and eta.
        k_z = Power_spec.eta_to_kz(self.eta,self.z,self.cosmo) # [Mpc^-1 h]
        k_x = Power_spec.uv_to_kxky(self.u_arr,self.z,self.cosmo) # [Mpc^-1 h]
        k_y = Power_spec.uv_to_kxky(self.v_arr,self.z,self.cosmo) # [Mpc^-1 h]

        # Creating 3D k_r array.
        self.k_r_arr = np.array([np.sqrt(k_x**2 + k_y**2 + kz**2) for kz in k_z]).T

        if wedge_cond:
            # Condition for ignoring the wedge contribution to the power spectrum.
            
            k_perp = np.sqrt(k_x**2 + k_y**2) # [Mpc^-1 h]
            
            grad_max = 0.5*np.pi # Calculating the max wedge cut. Accounting for window tapering.

            #wedge_cut = Power_spec.wedge_factor(self.z,self.cosmo) # Nicholes horizon cosmology cut.
            wedge_cut = grad_max*Power_spec.wedge_factor(self.z,self.cosmo) # Nicholes horizon cosmology cut.
            print('wedge_cut %5.3f' % wedge_cut)
            print('wedge_cut %5.3f' % (wedge_cut/grad_max))

            wedge_ind_cube = np.array([k_par < wedge_cut*k_perp for k_par in k_z]).T
            #wedge_ind_cube = np.array([k_par < fov*wedge_cut*k_perp for k_par in k_z]).T

            ## Testing
            #kr_min = 0.07
            kr_min = 0.1
            wedge_ind_cube[:,:,k_z < kr_min] = True
            ## 

            # Setting the wedge to zero.
            self.power_cube[wedge_ind_cube] = np.NaN
            self.weights_cube[wedge_ind_cube] = np.NaN
            self.k_r_arr[wedge_ind_cube] = np.NaN

            #kr_min = np.nanmin(self.k_r_arr[self.k_r_arr > 0.0])
            kr_max = np.nanmax(self.k_r_arr)

        else:

            kr_min = np.nanmin(self.k_r_arr[self.k_r_arr > 0.0])
            kr_max = np.nanmax(self.k_r_arr)

        # bin size is important. Too many bins, and some will have a sum of zero weights.
        #N_bins = 100 # This number provides integer bins sizes.
        #N_bins = 90 # This number provides integer bins sizes.
        N_bins = 60 # This number provides integer bins sizes. 
        
        print('k_r_min = %5.3f' % kr_min)
        print('k_r_max = %5.3f' % kr_max)

        # Log-linear bins.
        log_kr_min = np.log10(kr_min)
        log_kr_max = np.log10(kr_max)

        # Increments.
        dlog_k = (log_kr_max - log_kr_min)/N_bins
        dk = (kr_max - kr_min)/N_bins

        print('dk = %5.3f' % dk)

        #k_r_bins = np.logspace(log_kr_min - dlog_k/2,log_kr_max + dlog_k/2,N_bins + 1)
        k_r_bins = np.linspace(kr_min - dk/2,kr_max + dk/2,N_bins + 1)

        Power_spec1D = np.zeros(len(k_r_bins)-1)
        kr_vec = np.zeros(len(k_r_bins)-1)

        for i in range(len(k_r_bins)-1):

            # Calculating the radius:
            kr_vec[i] = ((k_r_bins[i+1] + k_r_bins[i])/2.0)
            #kr_vec[i] = 10**(0.5*(np.log10(k_r_bins[i+1]) + np.log10(k_r_bins[i])))

            # Defining the shell array index:
            shell_ind = np.logical_and(self.k_r_arr >= k_r_bins[i], self.k_r_arr <= k_r_bins[i+1])

            if wedge_cond:
                Power_spec1D[i] = np.average(self.power_cube[shell_ind],weights=self.weights_cube[shell_ind])
            else:
                Power_spec1D[i] = np.average(self.power_cube[shell_ind],weights=self.weights_cube[shell_ind])

        # Cosmological unit conversion factor:
        dnu = self.dnu*1e+6 #[Hz] full bandwidth.
        dnu_f = self.dnu_f*1e+6 #[Hz] fine channel width.
        Cosmo_factor = Power_spec.Power2Tb(dnu,dnu_f,self.nu_o,self.z,self.cosmo)

        self.Power1D = Power_spec1D*Cosmo_factor # [mK^3 Mpc^3 h^-3]
        self.k_r = kr_vec


    def Cylindrical(self,kb=kb,nu_21=nu_21,c=c):
        """
        Calculates the 2D cylindrical power spectra using the input Power object.
                
        Parameters
            ----------
            self : object
                Power object contains u and v arrays, as well as the observation redshift.
            kb : float
                Boltzman's constant.
            nu_21 : float
                21cm frequency in Hz.
            kb : float
                Speed of light km/s.
            
            Returns
            -------
        """

        bin_width = 2.5 # [lambda]
        
        # Converting into cosmological values.
        dk_r = Power_spec.uv_to_kxky(bin_width,self.z,self.cosmo) # h Mpc^-1
        k_perp_max = Power_spec.uv_to_kxky(self.uvmax,self.z,self.cosmo) # h Mpc^-1
        
        # Defininf the number of bins.
        N_bins = int(k_perp_max/dk_r)
        
        # Specifying the radius vector:
        kr_bins = np.linspace(0,k_perp_max,N_bins + 1)

        # Problems with instrument sampling. Ommiting first bin.
        kr_bins = kr_bins[1:]

        # The u_arr and v_arr should be shifted. 
        r_uv = np.sqrt(self.u_arr**2 + self.v_arr**2)
        kr_uv_arr = Power_spec.uv_to_kxky(r_uv,self.z,self.cosmo)

        # Initialising the power spectrum and radius arrays.
        Power_spec2D = np.zeros([len(self.eta),len(kr_bins)-1])
        kr_vec = np.zeros(len(kr_bins)-1)

        # Averaging the power in annular rings for each eta slice.
        for i in range(len(self.eta)):
            for j in range(len(kr_bins)-1):
                
                # Assigning the radius values. Needed for plotting purposes.
                kr_vec[j] = ((kr_bins[j+1] + kr_bins[j])/2.0)

                # Creating annunuls of boolean values.
                temp_ind = np.logical_and(kr_uv_arr >= kr_bins[j], kr_uv_arr <= kr_bins[j+1])

                # Weighted averaging of annuli values.
                Power_spec2D[i,j] = np.average(self.power_cube[temp_ind,i],weights=self.weights_cube[temp_ind,i]) ## Correct one.

        # Cosmological unit conversion factor:
        dnu = self.dnu*1e+6 #[Hz] full bandwidth.
        dnu_f = self.dnu_f*1e+6 #[Hz] fine channel width.
        Cosmo_factor = Power_spec.Power2Tb(dnu,dnu_f,self.nu_o,self.z,self.cosmo)

        # Assigning the power.
        self.Power2D = Power_spec2D*Cosmo_factor # [mK^3 Mpc^3 h^-3]

        # Assigning the perpendicular and parallel components of the power spectrum.
        self.kperp = kr_vec
        self.kpar = Power_spec.eta_to_kz(self.eta,self.z,self.cosmo)

    @staticmethod
    def plot_spherical(k_r,Power1D,figsize=(8,6),xlim=None,ylim=None,title=None,figaxs=None,\
        xlabel=None,ylabel=None,step=True,**kwargs):
        """
        Plot the 1D angular averaged power spectrum. If figaxs is provided allows for plotting
        more than one power spectrum.

            Parameters
            ----------
            k_r : numpy array, float
                1D vector of spherically radial k-modes.
            Power1D : numpy array, float
                1D Power.
            
            Returns
            -------
            None
        """

        if figaxs:
            fig = figaxs[0]
            axs = figaxs[1]
        else:
            fig, axs = plt.subplots(1, figsize = figsize, dpi=75)

        plt.loglog()

        if step:
            # Default options is a step plot.
            axs.step(k_r,Power1D,**kwargs)
        else:
            # Line plot is more useful for comparing Power spectra with different bin sizes.
            axs.plot(k_r,Power1D,**kwargs)


        if xlim:
            axs.set_xlim(xlim)
        if ylim:
            axs.set_ylim(ylim)

        if xlabel:
            axs.set_xlabel(xlabel,fontsize=24)
        else:
            axs.set_xlabel(r'$|\mathbf{k}| \,[\rm{h\,Mpc^{-1}}]$',fontsize=24)

        if ylabel:
            axs.set_ylabel(ylabel,fontsize=24)
        else:
            axs.set_ylabel(r'$\rm{P(\mathbf{k}) \, [mK^2\,h^{-3}\,Mpc^3]}$',fontsize=24)

        axs.tick_params(axis='x',labelsize=20)
        axs.tick_params(axis='y',labelsize=20)

        axs.grid(False)

        if figaxs:
            if title:
                plt.savefig('{0}.png'.format(title))
            return axs
            
        else:
            plt.tight_layout()
        
            if title:
                plt.savefig('{0}.png'.format(title))
            else:
                plt.show()

    @staticmethod
    def plot_cylindrical(Power2D,kperp,kpar,figsize=(7.5,10.5),cmap='viridis',
        name=None,xlim=None,ylim=None,vmin=None,vmax=None,clab=None,lognorm=True,title=None,**kwargs):

        """
        Plot the 2D cylindrically averaged power spectrum.

            Parameters
            ----------
            Power2D : numpy array, float
                2D numpy array containing the power.
            kperp : numpy array, float
                1D vector of perpendicular k-modes.
            kpar : numpy array, float
                1D vector of parallel k-modes.
            
            Returns
            -------
            None
        """

        fov = 0.076
        vol_fac = 0.35501

        fig, axs = plt.subplots(1, figsize = figsize, dpi=75, constrained_layout=True)

        if vmax:
            vmax=vmax
        else:
            pspec_max = np.max(np.log10(Power2D[Power2D > 0]))
            vmax = 10**pspec_max
        
        if vmin != None:
            # If vmin=0 this is considered false. So condition has to be different
            # to vmax.
            vmin=vmin
        else:
            pspec_min = np.min(np.log10(Power2D[Power2D > 0]))

            vmin = 10**pspec_min

        if lognorm:
            norm = matplotlib.colors.LogNorm()
            #norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
        else:
            norm = None

        print('Min = %5.3e' % np.min(Power2D[Power2D > 0]))
        print('Max = %5.3e' % np.max(Power2D[Power2D > 0].flatten()[0]))
        
        # Setting NaN values to a particular colour:
        cmap = matplotlib.cm.viridis
        cmap.set_bad('lightgray',1.)

        im = axs.imshow(Power2D,cmap=cmap,origin='lower',\
                extent=[np.min(kperp),np.max(kperp),np.min(kpar),np.max(kpar)],**kwargs,\
                    norm=norm,vmin=vmin,vmax=vmax, aspect='auto')
        

        # Setting the colour bars:
        cb = fig.colorbar(im, ax=axs, fraction=0.04, pad=0.002, extend='both')

        if clab:
            cb.set_label(label=clab,fontsize=20)
        else:
            cb.set_label(label=r'$P(k_\perp,k_{||}) \, [\rm{mK^2\,h^{-3}\,Mpc^3}]$',fontsize=20)
        
        axs.set_xscale('log')
        axs.set_yscale('log')

        ####
        from astropy.cosmology import Planck18

        # Cosmological scaling parameter:
        z = 7.14
        h = Planck18.H(0).value/100 # Hubble parameter.
        E_z = Planck18.efunc(z) ## Scaling function, see (Hogg 2000)

        # Cosmological distances:
        Dm = Planck18.comoving_distance(z).value*h #[Mpc/h] Transverse co-moving distance.
        DH = 3000 # [Mpc/h] Hubble distance.

        # Full sky.
        grad_max = 0.5*np.pi*Dm*E_z/(DH*(1 + z)) # Morales et al (2012) horizon cosmology cut.
        # Primary beam FOV.
        grad = np.sqrt(fov)*Dm*E_z/(DH*(1 + z)) # Morales et al (2012) horizon cosmology cut.

        # Illustrating the EoR window and horizon lines.
        axs.plot([0.1/grad_max,np.max(kperp)],grad_max*np.array([0.1/grad_max,np.max(kperp)]),lw=3,c='k')
        axs.plot([0.008,0.1/grad_max-0.0002839],[0.1,0.1],lw=3,c='k')
        axs.plot(kperp,grad*kperp,lw=3,ls='--',c='k')

        if xlim:
            axs.set_xlim(xlim)
        else:
            axs.set_xlim([0.008,np.max(kperp)])
            #axs.set_xlim([np.min(kperp),np.max(kperp)])
            
        if ylim:
            axs.set_ylim(ylim)
        else:
            axs.set_ylim([0.01,np.max(kpar)])

        axs.set_xlabel(r'$k_\perp \,[\rm{h\,Mpc^{-1}}]$',fontsize=20)
        axs.set_ylabel(r'$k_{||}\,[\rm{h\,Mpc^{-1}}]$',fontsize=20)

        # Setting the tick label fontsizes.
        axs.tick_params(axis='x', labelsize=18)
        axs.tick_params(axis='y', labelsize=18)
        cb.ax.tick_params(labelsize=18)

        axs.grid(False)

        if title:
            plt.title(title,fontsize=20)
        else:
            pass

        if name:
            plt.savefig('{0}.png'.format(name))
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
    
    def __init__(self,array_loc=array_loc,test_gauss=False,test_uniform=False):
        

        if test_gauss:
            # Randomly generated gaussian array.
            array_gauss = np.loadtxt('/home/jaiden/Documents/EoR/OSIRIS/gaussian_antenna_locations_MWA_phase1.txt')
            self.east = array_gauss[:,0] # [m]
            self.north = array_gauss[:,1] # [m]
            self.height = array_gauss[:,2] # [m]
        elif test_uniform:
            # Randomly generated uniform array. 
            array_uni = np.loadtxt('/home/jaiden/Documents/EoR/OSIRIS/uniform_antenna_locations_MWA_phase1.txt')
            self.east = array_uni[:,0] # [m]
            self.north = array_uni[:,1] # [m]
            self.height = array_uni[:,2] # [m]
        else:
            # Defualt case, uses the MWA Phase I layout.
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

        Author: J.Line
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
    
    def uvw_lam(self,wavelength,uvmax=None):
        """
        Converts the (u,v,w) coordinates from meters to wavelengths. Additionally
        subsets for the uvmax cooridinate.
        """
    
        # Converting into wavelengths.
        u_lam = self.u_m/wavelength 
        v_lam = self.v_m/wavelength
        w_lam = self.w_m/wavelength
        
        # Determining the uv_max boolean mask
        #uv_mask = (np.abs(u_lam) < uvmax)*(np.abs(v_lam) < uvmax)
        if uvmax:
            # If uvmax is given, otherwise return the entire array.
            uv_mask = np.logical_and(np.abs(u_lam) <= uvmax, np.abs(v_lam) <= uvmax)
        
            self.uv_mask = uv_mask # Useful for rephasing purposes.

            self.u_lam = u_lam[uv_mask]
            self.v_lam = v_lam[uv_mask]
            self.w_lam = w_lam[uv_mask]
        else:
            self.u_lam = u_lam
            self.v_lam = v_lam
            self.w_lam = w_lam
        
    def plot_arr(self,uvmax=None,figsize=(10,10)):
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

        if uvmax:
            ax1.set_xlim(-uvmax,uvmax)
            ax1.set_ylim(-uvmax,uvmax)
        else:
            pass

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
    add_point_sources(freqs=""):
        ...
    plot_sky_mod()
        
    """
    
    def __init__(self,shape,l_vec,m_vec):
        # Probably only need to give the (L,M) and shape. Can make the vectors in the initialisation.
        self.model = np.zeros(shape)
        
        self.l_vec = l_vec
        self.m_vec = m_vec
        
        # Useful for specifying the minimum Gaussian sizes.
        # Small Gaussians should still be sampled by several pixels. This effectively
        # creates a psf.
        self.dl = np.abs(l_vec[1]-l_vec[0])/2
        self.dm = np.abs(m_vec[1]-m_vec[0])/2

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
        Az_arr[self.ind_arr] = 2*np.pi -  (np.arctan2(self.l_grid[self.ind_arr],self.m_grid[self.ind_arr]) + np.pi) #arctan2() returns [-pi,pi] we want [0,2pi].
        #Az_arr[self.ind_arr] = np.arctan2(self.l_grid[self.ind_arr],self.m_grid[self.ind_arr]) + np.pi #arctan2() returns [-pi,pi] we want [0,2pi].
        
        # Defining the Altitude and Azimuthal grids.
        self.Alt_grid = Alt_arr
        self.Az_grid = Az_arr
        
    

    def Gauss2D(self,Az,Zen,Sint,Az0,Zen0,theta_pa,amaj,bmin):
        """
        Generates 2D Gaussian array.

        Parameters
        ----------
        Az : numpy array, float
            2D azimuth numpy array. [rad]
        Az0 : numpy array, float
            Azimuth angle of the Gaussian centre. [rad]
        Zen : numpy array, float
            2D zenith numpy array. [rad]
        Zen0 : numpy array, float
            Zenith angle of the centre of the Gaussian. [rad]
        amaj : numpy array, float
            Gaussian major axis. [deg]
        bmin : numpy array, float
            Gaussian minor axis. [deg]
        theta_pa : numpy array, float
            Gaussian position angle. [rad]
        Sint : numpy array, float
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

        # Defining the width of the Gaussians
        sigx = amaj/(2.0*np.sqrt(2.0*np.log(2.0)))
        sigy = bmin/(2.0*np.sqrt(2.0*np.log(2.0)))

        # Adjusting for offset position from zenith. Accounting for projection effects.
        sigx = sigx*np.sqrt((np.sin(theta_pa))**2 + (np.cos(theta_pa)*np.cos(Zen0))**2)
        sigy = sigy*np.sqrt((np.cos(theta_pa))**2 + (np.sin(theta_pa)*np.cos(Zen0))**2)

        # Checking to see if the new widths are smaller than the pixel sampling scale.
        if sigx < self.dl:
            # If smaller then set the minimum size to be the quadrature sum of the smallest scale,
            # and the new sigma x.
            sigx = np.sqrt(self.dl**2 + sigx**2)
        else:
            pass

        # Checking to see if the new widths are smaller than the pixel sampling scale.
        if sigy < self.dl:
            # If smaller then set the minimum size to be the quadrature sum of the smallest scale,
            # and the new sigma y.
            sigy = np.sqrt(self.dl**2 + sigy**2)
        else:
            pass

        # Deriving the peak amplitude from the integrated amplitude.
        Speak = Sint/(sigx*sigy*2*np.pi)

        #theta = theta_pa + Az0
        theta = theta_pa
        
        a = (np.cos(theta)**2)/(2.0*sigx**2) + (np.sin(theta)**2)/(2.0*sigy**2)
        b = -np.sin(2.0*theta)/(4.0*sigx**2) + np.sin(2.0*theta)/(4.0*sigy**2)    
        c = (np.sin(theta)**2)/(2.0*sigx**2) + (np.cos(theta)**2)/(2.0*sigy**2)

        x_shft = np.sin(Zen)*np.cos(Az) - np.sin(Zen0)*np.cos(Az0)
        y_shft = -np.sin(Zen)*np.sin(Az) + np.sin(Zen0)*np.sin(Az0)
        
        return Speak*np.exp(-(a*(x_shft)**2 + 2*b*(x_shft)*(y_shft) + c*(y_shft)**2))
    
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
        self.m_mod = np.cos(np.radians(Alt_mod))*np.cos(np.radians(Az_mod))# Slant Orthographic Project
        #self.m_mod = -np.cos(np.radians(Alt_mod))*np.cos(np.radians(Az_mod))# Slant Orthographic Project

        if np.shape(self.l_mod):
            n_sources = len(self.l_mod)
        else:
            n_sources = 1

        #for i in range(n_sources):
        for i in tqdm(range(n_sources)):

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
            #Az_temp_arr = np.arctan2(m_temp_arr,l_temp_arr) + np.pi  #arctan2() returns [-pi,pi] we want [0,2pi].
            Az_temp_arr = 2*np.pi - (np.arctan2(l_temp_arr,-m_temp_arr) + np.pi)  #arctan2() returns [-pi,pi] we want [0,2pi].

            # converting the major and minor axes into (l,m) coords.
            if np.shape(self.l_mod):
                # Multiple source case where shape(l_mod) is not None type.
                temp_maj = np.radians(Maj[i])
                temp_min = np.radians(Min[i])
        
                Gauss_temp = self.Gauss2D(Az_temp_arr, np.pi/2 - Alt_temp_arr, 1.0, 2*np.pi - np.radians(Az_mod[i]),\
                                np.pi/2 - np.radians(Alt_mod[i]),np.radians(PA[i]),\
                                temp_maj, temp_min)
            
                # Creating temporary array which has dimensions of (l,m,frequency).
                Gauss_temp_arr = np.ones(self.model[l_ind_arr,m_ind_arr,:].shape)*Gauss_temp[:,:,None]

                # scaling the array by the integrated frequency dependent flux density, and adding to the model.
                self.model[l_ind_arr,m_ind_arr,:] += S[i,:]*Gauss_temp_arr

            else:
                # Single source case.
                temp_maj = np.radians(Maj)
                temp_min = np.radians(Min)

                Gauss_temp = self.Gauss2D(Az_temp_arr, np.pi/2 - Alt_temp_arr, 1.0, 2*np.pi - np.radians(Az_mod),\
                                np.pi/2 - np.radians(Alt_mod),np.radians(PA),\
                                temp_maj, temp_min)

                # Creating temporary array which has dimensions of (l,m,frequency).
                Gauss_temp_arr = np.ones(self.model[l_ind_arr,m_ind_arr,:].shape)*Gauss_temp[:,:,None]

                # scaling the array by the integrated frequency dependent flux density, and adding to the model.
                self.model[l_ind_arr,m_ind_arr,:] += S*Gauss_temp_arr

            ## Set all NaNs and values below the horizon to zero:
            #self.model[self.r_arr > 1.0,:] = 0.0
            self.model[np.isnan(self.model)] = 0.0

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
        self.m_mod = np.cos(np.radians(Alt_mod))*np.cos(np.radians(Az_mod))# Slant Orthographic Project
        #self.m_mod = -np.cos(np.radians(Alt_mod))*np.cos(np.radians(Az_mod))# Slant Orthographic Project

        # For the point source location.
        L = (np.max(self.l_vec) - np.min(self.l_vec))

        N = len(self.l_vec)
        dA = self.dl*self.dm

        if np.shape(self.l_mod):
            # Multiple source case.
            n_sources = len(self.l_mod)
        else:
            # Single source case.
            n_sources = 1

        for i in range(n_sources):

            # Creating temporary close l and m mask arrays:
            if np.shape(self.l_mod):
                # Multiple source case where shape(l_mod) is not None type.
                l_ind, m_ind = find_closest_xy(self.l_mod[i],self.m_mod[i],self.l_vec,self.m_vec)
            else:
                # Single source case.
                l_ind, m_ind = find_closest_xy(self.l_mod,self.m_mod,self.l_vec,self.m_vec)

            # Setting point source value:
            #self.model[m_ind, l_ind,:] = 1.0#/dA
            self.model[m_ind, l_ind,:] = S[i]/dA

            ## Set all NaNs and values below the horizon to zero:
            #self.model[np.isnan(self.model)] = 0.0

            #if np.shape(self.l_mod):
            #    # Multiple source case:
            #    #self.model = self.model*S[i,:]
            #    self.model = self.model*S[i]
            #else:
            #    # Default single source case.
            #    self.model = self.model*S

    def plot_sky_mod(self,window=None,index=None,figsize=(14,14),xlab=r'$l$',ylab=r'$m$',cmap='cividis',\
        clab=None,title=None,vmax=None,vmin=None,lognorm=False,**kwargs):
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

        # These values should be unpacked with kwargs.
        if lognorm:
            norm = matplotlib.colors.LogNorm()
        else:
            norm = None

        if np.any(vmax):
            vmax = vmax
        else:
            vmax = None
        
        if np.any(vmin):
            vmin = vmin
        else:
            vmin = None

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

            if len(self.model[0,0,:]) > 100:
                im = axs.imshow(self.model[l_ind_arr,m_ind_arr,100],cmap=cmap,origin='lower',\
                    extent=[np.min(l_temp_arr),np.max(l_temp_arr),np.min(m_temp_arr),np.max(m_temp_arr)],\
                    vmin=vmin,vmax=vmax,aspect='auto')
            else:
                im = axs.imshow(self.model[l_ind_arr,m_ind_arr,0],cmap=cmap,origin='lower',\
                    extent=[np.min(l_temp_arr),np.max(l_temp_arr),np.min(m_temp_arr),np.max(m_temp_arr)],\
                    vmin=vmin,vmax=vmax,aspect='auto')
        else:
            if len(self.model[0,0,:]) > 100:
                # Case for the whole sky.
                temp_arr = np.ones(self.model[:,:,0].shape)*self.model[:,:,100]
                temp_arr[self.r_grid > 1.0] = np.NaN

                im = axs.imshow(temp_arr,cmap=cmap,origin='lower',\
                    extent=[np.min(self.l_grid),np.max(self.l_grid),np.min(self.m_grid),np.max(self.m_grid)],\
                    vmin=vmin,vmax=vmax,aspect='auto')
            else:
                temp_arr = self.model[:,:,0]
                temp_arr[self.r_grid > 1.0] = np.NaN

                im = axs.imshow(temp_arr,cmap=cmap,origin='lower',\
                    extent=[np.min(self.l_grid),np.max(self.l_grid),np.min(self.m_grid),np.max(self.m_grid)],\
                    vmin=vmin,vmax=vmax,aspect='auto')

        if clab:
            # Find a better way to do this.
            clab = clab
        else:
            # Default colour bar label.
            clab = r'$I_{\rm{app}}\,[\rm{Jy/Str}]$'

        # Setting the colour bars:
        if np.any(vmax) and np.any(vmin):
            # Setting the limits of the colour bar. 
            cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, extend='both')
        elif np.any(vmax):
            # Upper limit only.
            cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, extend='max')
        elif np.any(vmin):
            # Lower limit only.
            cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, extend='min')
        else:
            # No limits.
            cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04)

        cb.set_label(label=clab,fontsize=24)

        axs.set_xlabel(xlab,fontsize=24)
        axs.set_ylabel(ylab,fontsize=24)

        im.set_clim(**kwargs)

        if title:
            plt.savefig('{0}.png'.format(title))
        else:
            plt.show()