#!/usr/bin/python

__author__ = "Jaiden Cook, Jack Line"
__credits__ = ["Jaiden Cook","Jack Line"]
__version__ = "1.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"

# Generic stuff:
import time
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

# Array stuff:
import numpy as np
warnings.simplefilter('ignore', np.RankWarning)

# Plotting stuff:
import matplotlib.pyplot as plt
import matplotlib

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


# Scipy stuff:
import scipy
from scipy.fft import ifftn,fftn,fftfreq,fftshift,ifftshift


# MWA beam stuff
from mwa_pb import primary_beam as pb

def mwa_alt_az_za(obsid, ra=None, dec=None, degrees=False):
    """
    Calculate the altitude, azumith and zenith for an obsid
    Parameters
    ----------
    obsid : float
        The MWA observation id (GPS time)
    ra : float 
        The right acension in HH:MM:SS
    dec : float
        The declintation in HH:MM:SS
    degrees : Bool 
        If true the ra and dec is given in degrees (Default:False)
    
    Returns
    -------
    Alt : float
        Altitude angle in radians or degrees.
    Az : float
        Azimuth angle in radians or degrees.
    Za : float
        Zenith angle in radians or degrees.
    """

    #
    ## This function should be moved to the MWA_array class as a static function
    #

    from astropy.time import Time
    from astropy.coordinates import SkyCoord, AltAz, EarthLocation
    from astropy import units as u
   
    obstime = Time(float(obsid),format='gps')
   
    if degrees:
        sky_posn = SkyCoord(ra, dec, unit=(u.deg,u.deg))
    else:
        sky_posn = SkyCoord(ra, dec, unit=(u.hourangle,u.deg))
    
    earth_location = EarthLocation.of_site('Murchison Widefield Array')
    # Manual coordinates for MWA location.
    #earth_location = EarthLocation.from_geodetic(lon="116:40:14.93", lat="-26:42:11.95", height=377.8)
    
    altaz = sky_posn.transform_to(AltAz(obstime=obstime, location=earth_location))
    
    Alt = altaz.alt.deg
    Az = altaz.az.deg
    Za = 90. - Alt
    return Alt, Az, Za
       
def Gauss2D(X,Y,A,x0,y0,theta,amaj,bmin,polar=False):
    """
    Generates 2D Gaussian array.

    Parameters
    ----------
    x : numpy array, float
        2D cartesian or azimuth numpy array. [rad]
    y : numpy array, float
        2D cartesian or zenith numpy array. [rad]
    x0 : numpy array, float
        Cartesian or Azimuth angle of the Gaussian centre. [rad]
    y0 : numpy array, float
        Cartesian or Zenith angle of the centre of the Gaussian. [rad]
    amaj : numpy array, float
        Gaussian major axis. [deg]
    bmin : numpy array, float
        Gaussian minor axis. [deg]
    theta : numpy array, float
        Gaussian position angle. [rad]
    Sint : numpy array, float
        Source integrated flux density.

    Returns
    -------
    2D Gaussian array.
    """
    ###
    ### This should be removed in favour of the sky-model version
    ### Keep this might be used in modelling of sources. 

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
        # 
        # https://www.aanda.org/articles/aa/full/2002/45/aah3860/node5.html
        #
    
        # A*exp(-(a*(x-x0)^2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)^2))
        #
        # x = sin(Zen)*cos(Az) - sin(Zen0)*cos(Az0)
        # y = -sin(Zen)*sin(Az) + sin(Zen0)*sin(Az0)
        #
        # Zen in [0,pi/2]
        # Az in [0,2pi]
    

        #l0 = np.sin(x0)*np.cos(y0)
        #m0 = -np.sin(x0)*np.sin(y0)

        Az = X
        Zen = Y
        Az0 = x0
        Zen0 = y0
        theta_pa = theta

        # In the polar case the size of the Gaussian changes depending on zen0 and theta.
        # This calculates an approximation of the new size. See Cook et al (2022).        
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

def Plot_img(Img,X_vec=None,Y_vec=None,projection='cartesian',cmap='cividis',figsize = (7,7),\
    figaxs=None, xlab=r'$l$',ylab=r'$m$',clab='Intensity',lognorm=False,title=None,\
    clim=None,vmin=None,vmax=None,contours=None,savefig=False,**kwargs):
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

    #cmap = matplotlib.cm.viridis
    #cmap.set_bad('lightgray',1.)

    if projection == 'cartesian':

        if figaxs:
            fig = figaxs[0]
            axs = figaxs[1]
        else:
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
            if vmax > 1000:
                # Specifying formatting for large numbers.
                cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, format='%.1e',extend='max')
            else:
                cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04,extend='max')
        else:
            if np.nanmax(Img) > 1000:
                # Specifying formatting for large numbers. 
                cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, format='%.1e')
            else:
                # Don't use scientific notation if max is less than 1000.
                cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04)

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
        plt.title('{0}'.format(title),fontsize=22)
        if savefig:
            plt.savefig('{0}'.format(title))
        else:
            pass
    else:
        pass
        #plt.show()

def Plot_3D(X_arr,Y_arr,Z_arr,cmap='viridis',figsize=(7,7)):
    """
    Generates a 3D plot from an input (x,y) meshgrid, with a corresponding
    z-grid array. Default colourmap is 'viridis'.
    
    Parameters
    ----------
    X_arr : numpy array
        2D grid array of x values.
    Y_arr : numpy array
        2D grid array of y values.
    Z_arr : numpy array
        2D grid array of z values.
    cmap : string
        Name of the colour map. Must meet matplotlib requirements.
    figsize : tuple
        Figure size, default is (7,7).

    Returns
    -------
    None
    """
    fontsize=24
    fig = plt.figure(figsize = figsize, dpi=75)
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X_arr, Y_arr, Z_arr, cmap=cmap,
                       linewidth=0, antialiased=False)
    
    cb = fig.colorbar(surf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label='Intensity',fontsize=fontsize)

    ax.set_xlabel(r'$l$')
    ax.set_ylabel(r'$m$')
    
    plt.show()
    
def Plot_visibilites(Vis,u_vec,v_vec,cmap='viridis',lognorm=True,figsize=(7,7)):
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
                         extent=[np.min(u_vec),np.max(u_vec),np.min(v_vec),np.max(v_vec)])
    im_Real = axs[0,1].imshow(Vis.real,cmap=cmap,norm=norm,\
                          extent=[np.min(u_vec),np.max(u_vec),np.min(v_vec),np.max(v_vec)])
    im_Im = axs[1,0].imshow(Vis.imag,cmap=cmap,norm=norm,\
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
            Norm condition, accepts 'forward', 'backward', and 'ortho', 
            default is 'backward'.

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
    Vis = np.roll(np.roll(ifftshift(fftn(np.roll(np.roll(fftshift(img),
                    1,axis=0),1,axis=1),norm=norm)),-1,axis=0),-1,axis=1)
    #Vis = np.roll(np.roll(ifftshift(ifftn(np.roll(np.roll(fftshift(img),
    #               1,axis=0),1,axis=1),norm=norm)),-1,axis=0),-1,axis=1)

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
    """
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
    """
    ##
    ## Move this to Osiris_grid.py. Makes more sense to be there. 
    ##

    u_bit = (u_arr - u_cent)/sig_u
    v_bit = (v_arr - v_cent)/sig_v

    amp = 1/(2*np.pi*sig_u*sig_v)
    gaussian = amp*np.exp(-0.5*(u_bit**2 + v_bit**2))

    # Note that sum(gaussian)*dA = 1 thus sum(gaussian) = 1/dA.
    # The integral of Gaussian is what is equal to 1. int ~ sum*dA

    return gaussian

def find_closest_xy(x,y,x_vec,y_vec,off_cond=False):
    """
    Finds the indices for the (x,y) point associated to a x, y grid.

    Author : J. Line
    Modified by J. Cook

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
    """
    
    ##Find the difference between the gridded u coords and the desired u
    x_offs = np.abs(x_vec - x)
    y_offs = np.abs(y_vec - y)

    x_res = np.abs(x_vec[1]-x_vec[0])

    x_ind = np.argmin(x_offs)
    y_ind = np.argmin(y_offs)

    if np.any(x_offs[x_ind] > x_res/2) or np.any(y_offs[y_ind] > x_res/2):

        print('x and y coords out of axis bounds.')

        if off_cond:
            return None,None,None
        else:
            return None,None

    if off_cond:
        # Offset condition, if true return the offsets alongside the indices.
        x_offs = x - x_vec[x_ind] 
        y_offs = y - y_vec[y_ind]

        return x_ind, y_ind, (x_offs, y_offs)
    else:
        # Default condition don't return the offsets.
        return x_ind,y_ind

### Defining classes. Split this up into different module files.\


class MWA_uv:
    """
    Class defines the (u,v,w) coordinates for the MWA Phase I array in 
    terms of wavelengths. It take in different pointing angles. The default
    is a zenith pointing.
    ...

    Attributes
    ----------
        ...


    Methods
    -------
    enh2xyz(self,lat=MWA_lat):
        ...
    get_uvw(self,HA=H0,dec=MWA_lat)
        ...
    uvw_lam(self,wavelength,uvmax=None)
        ...
    plot_arr(self,uvmax=None,figsize=(10,10))
    """
    # Future can make it so you pass the RA and DEC of the phase centre.
    
    ## MWA latitude.
    MWA_lat = -26.703319444 # [deg] 
    ## Zenith hour angle.
    H0 = 0.0 # [deg]
    ## Array east, north, height data.
    path = "/home/jaiden/Documents/EoR/OSIRIS/data/"
    array_loc = np.loadtxt(f'{path}antenna_locations_MWA_phase1.txt')
    
    def __init__(self,array_loc=array_loc,test_gauss=False,test_uniform=False,path=path):
        
        if test_gauss:
            # Randomly generated gaussian array.
            array_gauss = np.loadtxt(f'{path}gaussian_antenna_locations_MWA_phase1.txt')
            self.east = array_gauss[:,0] # [m]
            self.north = array_gauss[:,1] # [m]
            self.height = array_gauss[:,2] # [m]
        elif test_uniform:
            # Randomly generated uniform array. 
            array_uni = np.loadtxt(f'{path}uniform_antenna_locations_MWA_phase1.txt')
            self.east = array_uni[:,0] # [m]
            self.north = array_uni[:,1] # [m]
            self.height = array_uni[:,2] # [m]
        else:
            # Defualt case, uses the MWA Phase I layout.
            self.east = array_loc[:,0] # [m]
            self.north = array_loc[:,1] # [m]
            self.height = array_loc[:,2] # [m]
        
        
    def enh2xyz(self,lat=MWA_lat):
        """
        Calculates local X,Y,Z using east,north,height coords,
        and the latitude of the array. Latitude must be in radians
        
        Author: J.Line

        Parameters
        ----------
        lat : float, default=MWA_lat
            Latitude of the array. Must be in radians.

        Returns
        -------
        """
        lat = np.radians(lat)
        
        sl = np.sin(lat)
        cl = np.cos(lat)
        self.X = -self.north*sl + self.height*cl
        self.Y = self.east
        self.Z = self.north*cl + self.height*sl
        
    def get_uvw(self,HA=H0,dec=MWA_lat):
        """
        Returns the (u,v,w) coordinates for a given pointing centre and hour angle.
        The default is a zenith pointing.

        Author: J.Line

        Parameters
        ----------
        dec : float, default=MWA_lat
            Declination of the observation. Default is MWA_lat which indicates a 
            zenith pointed observation.

        Returns
        -------
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

        # Filtering baselines with length less than 300 lambda.
        r_lam = np.sqrt(u_lam**2 + v_lam**2)
        
        # Determining the uv_max boolean mask
        #uv_mask = (np.abs(u_lam) < uvmax)*(np.abs(v_lam) < uvmax)
        if uvmax:
            # If uvmax is given, otherwise return the entire array.
            #uv_mask = np.logical_and(np.abs(u_lam) <= uvmax, np.abs(v_lam) <= uvmax)
            uv_mask = r_lam <= uvmax
        
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
        #self.dl = np.abs(l_vec[1]-l_vec[0])/2 # Old as of 29/7/2022
        self.dl = np.abs(l_vec[1]-l_vec[0])
        #self.dm = np.abs(m_vec[1]-m_vec[0])/2
        self.dm = np.abs(m_vec[1]-m_vec[0])

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
        Az_arr[self.ind_arr] = 2*np.pi - (np.arctan2(self.l_grid[self.ind_arr],self.m_grid[self.ind_arr]) + np.pi) #arctan2() returns [-pi,pi] we want [0,2pi].
        #Az_arr[self.ind_arr] = np.arctan2(self.l_grid[self.ind_arr],self.m_grid[self.ind_arr]) + np.pi #arctan2() returns [-pi,pi] we want [0,2pi].
        
        # Defining the Altitude and Azimuthal grids.
        self.Alt_grid = Alt_arr
        self.Az_grid = Az_arr
        
        self.l_mod = None
        self.m_mod = None
    
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
        #if sigx < self.dl: # Old as of 29/07/2022
        if sigx < self.dl/2:
            # If smaller then set the minimum size to be the quadrature sum of the smallest scale,
            # and the new sigma x.
            #sigx = np.sqrt(self.dl**2 + sigx**2)
            sigx = np.sqrt(0.25*self.dl**2 + sigx**2)
        else:
            pass

        # Checking to see if the new widths are smaller than the pixel sampling scale.
        #if sigy < self.dl: #Old as of 29/07/2022
        if sigy < self.dl/2: #Old as of 29/07/2022
            # If smaller then set the minimum size to be the quadrature sum of the smallest scale,
            # and the new sigma y.
            #sigy = np.sqrt(self.dl**2 + sigy**2)
            sigy = np.sqrt(0.25*self.dl**2 + sigy**2)
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
        #L = (np.max(self.l_vec) - np.min(self.l_vec))

        N = len(self.l_vec)
        #dA = self.dl*self.dm
        dA = 1/N**2#self.dl*self.dm

        if np.shape(self.l_mod):
            # Multiple source case.
            n_sources = len(self.l_mod)
        else:
            # Single source case.
            print('Source position (l,m) = (%5.2f,%5.2f)' % (self.l_mod,self.m_mod))
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

    def plot_sky_mod(self,window=None,index=None,figsize=(14,14),
                     xlab=r'$l$',ylab=r'$m$',cmap='cividis',clab=None,title=None,
                     vmax=None,vmin=None,lognorm=False,**kwargs):
        """
        This function plots a subset of the sky-model. Particularly for a single source.
        The main purpose of the functions in this pipeline is to plot the visibilities 
        for a single source. Additionally there is an all-sky plotting option.
        """

        fig, axs = plt.subplots(1, figsize = figsize, dpi=75)

        if index and np.any(self.l_mod):
            # Case for a single source, when there is more than one model source.
            l_mod = self.l_mod[index]
            m_mod = self.m_mod[index]
        elif np.any(self.l_mod):
            # Case for a single source, when there is more than one model source.
            l_mod = self.l_mod
            m_mod = self.m_mod

        # These values should be unpacked with kwargs.
        if lognorm:
            norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
        else:
            norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)


        if window and np.any(self.l_mod):
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
                im = axs.imshow(self.model[l_ind_arr,m_ind_arr,100],cmap=cmap,origin='lower',
                    extent=[np.min(l_temp_arr),np.max(l_temp_arr),np.min(m_temp_arr),np.max(m_temp_arr)],
                    aspect='auto',norm=norm)
            else:
                im = axs.imshow(self.model[l_ind_arr,m_ind_arr,0],cmap=cmap,origin='lower',
                    extent=[np.min(l_temp_arr),np.max(l_temp_arr),np.min(m_temp_arr),np.max(m_temp_arr)],
                    aspect='auto',norm=norm)
        else:
            if len(self.model[0,0,:]) > 100:
                # Case for the whole sky.
                temp_arr = np.ones(self.model[:,:,0].shape)*self.model[:,:,100]
                temp_arr[self.r_grid > 1.0] = np.NaN

                im = axs.imshow(temp_arr,cmap=cmap,origin='lower',
                    extent=[np.min(self.l_grid),np.max(self.l_grid),np.min(self.m_grid),np.max(self.m_grid)],
                    aspect='auto',norm=norm)
            else:
                temp_arr = self.model[:,:,0]
                temp_arr[self.r_grid > 1.0] = np.NaN

                im = axs.imshow(temp_arr,cmap=cmap,origin='lower',
                    extent=[np.min(self.l_grid),np.max(self.l_grid),np.min(self.m_grid),np.max(self.m_grid)],
                    aspect='auto',norm=norm)

        if clab:
            # Find a better way to do this.
            clab = clab
        else:
            # Default colour bar label.
            clab = r'$I\,[\rm{Jy/Str}]$'

        # Setting the colour bars:
        if np.any(vmax) and np.any(vmin):
            # Setting the limits of the colour bar. 
            cb = fig.colorbar(im, ax=axs, fraction=0.046, 
                              pad=0.04, extend='both', aspect=30)
        elif np.any(vmax):
            # Upper limit only.
            cb = fig.colorbar(im, ax=axs, fraction=0.046, 
                              pad=0.04, extend='max', aspect=30)
        elif np.any(vmin):
            # Lower limit only.
            cb = fig.colorbar(im, ax=axs, fraction=0.046, 
                              pad=0.04, extend='min', aspect=30)
        else:
            # No limits.
            cb = fig.colorbar(im, ax=axs, fraction=0.046, 
                              pad=0.04, aspect=30)

        cb.set_label(label=clab,fontsize=24)

        axs.set_xlabel(xlab,fontsize=24)
        axs.set_ylabel(ylab,fontsize=24)

        im.set_clim(**kwargs)

        if title:
            plt.savefig('{0}.png'.format(title))
        else:
            plt.show()