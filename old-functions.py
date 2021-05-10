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


def Vis_Beam_Poly2D(U,V,dL,dM,l0,m0,*a):
    
    a = np.array(a).ravel() # Setting the beam parameters.
    
    vis = np.zeros(np.shape(U),dtype=complex) # Initialising the vis array.
    p = int(np.sqrt(len(a)) - 1) # Forcing type.
    
    # Shifting the U and V arrays.
    U = fftshift(U)
    V = fftshift(V)
    
    index = 0
    for r in range(p+1):
        for s in range(p+1):
            
            # u-component:
            FT_b_u = 0
            for n in range(p-r+1):
                temp_frac_u = ((-1)**n) * ( ((dL/2.0)**(p-r-n)) / ((2*np.pi*(U))**(n+1)) )
                temp_cos_u = np.cos((np.pi/2.0)*(3*(p-r-n) + 1) - np.pi*(U)*dL)
                
                temp_u = temp_frac_u*temp_cos_u
                
                FT_b_u = FT_b_u + temp_u
                    


            # Taking care of the discontinuities.
            if r==2:
                # Sinc function condition.
                FT_b_u[np.isinf(FT_b_u)] = dL/2
            
            if r==1:
                FT_b_u[np.isnan(FT_b_u)] = 0.0
            
            if r==0:
                FT_b_u[np.isnan(FT_b_u)] = -(dL**3)/12.0
                    
            cond_u = False
            if r == 2 and cond_u == True:
                
                print(np.max(U[0,:]),np.min(U[0,:]),FT_b_u[0,:][U[0,:]==0.0])
                print(dL)
                plt.clf()
                #plt.semilogy(U[0,:],FT_b_u[0,:])
                plt.plot(U[0,:],FT_b_u[0,:])
                plt.xlabel(r'$u$',fontsize=24)
                plt.ylabel(r'$\frac{\hat{b}^2_2(u)}{2i^{2}e^{-2\pi i u l_0}}$',fontsize=24)
                plt.xlim([-25,25])
                #plt.plot(U[0,:],temp_cos_u[0,:])
                #plt.savefig('{0}.png'.format(n))
                plt.show()
                
            # v-component:
            FT_b_v = 0
            for q in range(s+1):
                temp_frac_v = ((-1)**q)*(((dM/2.0)**(s-q))/(2*np.pi*(V))**(q+1))
                temp_cos_v = np.cos((np.pi/2.0)*(3*(s-q) + 1) - np.pi*(V)*dM)
                
                temp_v = temp_frac_v*temp_cos_v
                
                FT_b_v = FT_b_v + temp_v
                
                if s==0:
                    # Sinc function condition.
                    FT_b_v[np.isinf(FT_b_v)] = dM/2
                
                if s==1:
                    FT_b_v[np.isnan(FT_b_v)] = 0.0
                
                if s==2:
                    FT_b_v[np.isnan(FT_b_v)] = -(dM**3)/12.0
                
            cond_v = False
            if s == 2 and cond_v == True:
                
                print(np.max(V[:,0]),np.min(V[:,0]),FT_b_v[:,0][V[:,0]==0.0])
                print(dM)
                plt.clf()
                plt.plot(V[:,0],FT_b_v[:,0])
                plt.show()
            
            vis = vis + 4*(complex(0,1)**(p-r-s))*a[index]*FT_b_u*FT_b_v
            
            index = index + 1
            
    # Exponential phase offset term.
    phase_l0m0 = np.zeros(np.shape(U),dtype=complex)
    phase_l0m0.real = np.cos(-2*np.pi*(l0*U + m0*V))
    phase_l0m0.imag = np.sin(-2*np.pi*(l0*U + m0*V))
    
    # shifting the phase for the off centre window.
    vis = vis*phase_l0m0
    return vis
    

def Vis_Gauss2D(U,V,I0,l0,m0,PA,amaj,bmin,Az0,Zen0):
    
    # Analytic visibility model.
    
    Vis = np.zeros(np.shape(U),dtype=complex) # Initialising the vis array.

    N = len(Vis)
    #Normalisation = ((N**2)/(4*np.pi**2))
    Normalisation = (N/(2*np.pi))**2#N/(2*np.pi)

    
    # Defining the width of the Gaussians
    sigx = amaj/np.sqrt(2.0*np.log(2.0))
    sigy = bmin/np.sqrt(2.0*np.log(2.0))

    # Defining the width of the Gaussians
    sigx = amaj/(2.0*np.sqrt(2.0*np.log(2.0)))
    sigy = bmin/(2.0*np.sqrt(2.0*np.log(2.0)))

    
    sigx = sigx*np.sqrt((np.sin(PA))**2 + (np.cos(PA)*np.cos(Zen0))**2)
    sigy = sigy*np.sqrt((np.cos(PA))**2 + (np.sin(PA)*np.cos(Zen0))**2)
    
    #print('l0 = %2.3f, m0 = %2.3f' %(l0,m0))
    #print('PA = %2.3f' % (PA))

    #PA = PA - np.arctan2(l0,m0)+np.pi
    PA = PA + Az0

    #print('PA = %2.3f' % (PA))
    #print('arctan(l0,m0) = %2.3f' % (np.arctan2(l0,m0)))


    I0_h = 2*np.pi*sigx*sigy*I0


    a_p = 2*(np.pi**2)*((sigx**2)*(np.cos(PA)**2) + (sigy**2)*(np.sin(PA)**2))
    b_p = 0.5*(np.sin(2*PA))*(np.pi**2)*(-1*(-sigx**2) + (-sigy**2))
    c_p = 2*(np.pi**2)*((sigy**2)*(np.cos(PA)**2) + (sigx**2)*(np.sin(PA)**2))
    
    # Exponential phase offset term.
    phase_l0m0 = np.zeros(np.shape(U),dtype=complex)
    phase_l0m0.real = np.cos(-2*np.pi*(l0*U + m0*V))
    phase_l0m0.imag = np.sin(-2*np.pi*(l0*U + m0*V))
    
    # Normalisation issue

    #Fix normalisation issues. Determine how to properly normalise.
    Vis = Vis + I0_h*np.exp(-a_p*U**2 - 2*b_p*U*V - c_p*V**2)
    Vis = phase_l0m0*Vis*Normalisation #Normalisation = (N^2/4pi^2)
    
    return Vis

def Poly_func2D_old(xx,yy,*a):
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

        #print('l0 = %2.3f, m0 = %2.3f' %(l0,m0))


        #print(l0,m0,np.sin(np.arccos(np.sqrt(l0**2 + m0**2))),np.arccos(np.sqrt(l0**2 + m0**2)))

        #sigy = 2*np.sin(sigy/2)*np.sin(np.arccos(np.sqrt(l0**2 + m0**2)))#*np.cos(np.pi/2 - np.arctan2(l0,m0)+np.pi)#*np.cos(theta)
        #sigx = 2*np.sin(sigx/2)

        Az = X
        Zen = Y
        Az0 = x0
        Zen0 = y0
        theta_pa = theta

        #print('theta = %2.3f' % (theta))
        
        sigx = sigx*np.sqrt((np.sin(theta_pa))**2 + (np.cos(theta_pa)*np.cos(Zen0))**2)
        sigy = sigy*np.sqrt((np.cos(theta_pa))**2 + (np.sin(theta_pa)*np.cos(Zen0))**2)

        #theta = theta + np.arctan2(l0,m0) + np.pi
        theta = theta + Az0
        
        a = (np.cos(theta)**2)/(2.0*sigx**2) + (np.sin(theta)**2)/(2.0*sigy**2)
        b = -np.sin(2.0*theta)/(4.0*sigx**2) + np.sin(2.0*theta)/(4.0*sigy**2)    
        c = (np.sin(theta)**2)/(2.0*sigx**2) + (np.cos(theta)**2)/(2.0*sigy**2)
        
        #print('theta = %2.3f' % (theta))
        #print('arctan(l0,m0) = %2.3f' % (np.arctan2(m0,l0)))
        
        # Deriving the peak amplitude from the integrated amplitude.
        Amplitude = A/(sigx*sigy*2*np.pi)

        # Defining x-x0 and y-y0. Defining them in spherical coordinates.
        ##x_shft = 2*np.sin(Zen)*np.cos(Az)/(1+np.cos(Zen)) - 2*np.sin(Zen0)*np.cos(Az0)/(1+np.cos(Zen0))
        ##y_shft = 2*np.sin(Zen)*np.sin(Az)/(1+np.cos(Zen)) - 2*np.sin(Zen0)*np.sin(Az0)/(1+np.cos(Zen0))

        x_shft = np.sin(Zen)*np.cos(Az) - np.sin(Zen0)*np.cos(Az0)

        y_shft = -np.sin(Zen)*np.sin(Az) + np.sin(Zen0)*np.sin(Az0)

    
        return Amplitude*np.exp(-(a*(x_shft)**2 + 2*b*(x_shft)*(y_shft) + c*(y_shft)**2))

# Old Jack functions.

def get_uvw(x_lamb,y_lamb,z_lamb,dec,HA):
    '''Calculates u,v,w for a given 
    
    Author: J.Line
    '''

    u = np.sin(HA)*x_lamb + np.cos(HA)*y_lamb
    v = -np.sin(dec)*np.cos(HA)*x_lamb + np.sin(dec)*np.sin(HA)*y_lamb + np.cos(dec)*z_lamb
    w = np.cos(dec)*np.cos(HA)*x_lamb - np.cos(dec)*np.sin(HA)*y_lamb + np.sin(dec)*z_lamb
    return u,v,w

def enh2xyz(east,north,height,latitiude):
    '''Calculates local X,Y,Z using east,north,height coords,
    and the latitude of the array. Latitude must be in radians
    
    Author: J.Line
    '''
    sl = np.sin(latitiude)
    cl = np.cos(latitiude)
    X = -north*sl + height*cl
    Y = east
    Z = north*cl + height*sl
    return X,Y,Z

def MWA_uvw(MWA_lat=-26.7033194444,H0=0.0):
    """
    Returns the (u,v,w) coordinates for a given pointing centre and hour angle.
    The default is a zenith pointing.
    """
    
    ##Text file containing e,n,h coords.
    array_layout = 'antenna_locations_MWA_phase1.txt'
    anntenna_locs = np.loadtxt(array_layout)
    east, north, height = anntenna_locs[:,0],anntenna_locs[:,1],anntenna_locs[:,2]

    #MWA_lat = -26.7033194444
    #H0 = 0.0

    ##Do conversion from enh into XYZ
    X,Y,Z = enh2xyz(east, north, height, np.radians(MWA_lat))

    x_lengths = []
    y_lengths = []
    z_lengths = []

    # Calculating for each baseline.
    for tile1 in range(0,len(X)):
        for tile2 in range(tile1+1,len(X)):
            x_len = X[tile2] - X[tile1]
            y_len = Y[tile2] - Y[tile1]
            z_len = Z[tile2] - Z[tile1]
        
            x_lengths.append(x_len)
            y_lengths.append(y_len)
            z_lengths.append(z_len)

    # These are in metres not wavelengths.
    dx = np.array(x_lengths) # [m] 
    dy = np.array(y_lengths) # [m]
    dz = np.array(z_lengths) # [m]

    # These are the general (u,v,w) values. 
    u_m,v_m,w_m = get_uvw(dx,dy,dz,np.radians(MWA_lat),np.radians(H0)) #[m]

    return u_m, v_m, w_m

def uvw_lam(u_m,v_m,w_m,wave,uvmax):
    """
    Converts the (u,v,w) coordinates from meters to wavelengths. Additionally
    subsets for the uvmax cooridinate.
    """
    
    # Converting into wavelengths.
    u_lam = u_m/wave 
    v_lam = v_m/wave
    w_lam = w_m/wave
    
    # Determining the uv_max boolean mask
    uv_mask = (np.abs(u_lam) < uvmax)*(np.abs(v_lam) < uvmax)
    
    u_lam = u_lam[uv_mask]
    v_lam = v_lam[uv_mask]
    w_lam = w_lam[uv_mask]
    
    return u_lam, v_lam, w_lam

def Plot_MWA_uv(u_lam, v_lam, uvmax,figsize=(10,10)):
    """
    Plots the MWA uv sample for a max uv cutoff. Units are in wavelengths.
    """
    
    plt.clf()

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.plot(u_lam,v_lam,'k.',mfc='none',ms=1)
    #ax1.plot(-u_lam,-v_lam,'k.',mfc='none',ms=1)
    ax1.set_xlabel(r'$u\,(\lambda)$',fontsize=24)
    ax1.set_ylabel(r'$v\,(\lambda)$',fontsize=24)
    ax1.set_xlim(-uvmax,uvmax)
    ax1.set_ylim(-uvmax,uvmax)

    plt.show()

def add_kernel(uv_array,u_ind,v_ind,kernel):
    '''Takes v by u sized kernel and adds it into
    a numpy array at the u,v point u_ind, v_ind
    Kernel MUST be odd dimensions for symmetry purposes
    
    Author: J.Line

    '''
    ker_v,ker_u = kernel.shape
    width_u = int((ker_u - 1) / 2)
    width_v = int((ker_v - 1) / 2)

    N = len(uv_array)
    min_u_ind = u_ind - width_u
    max_u_ind = u_ind + width_u + 1
    min_v_ind = v_ind - width_v
    max_v_ind = v_ind + width_v + 1
    
    ## Jack suggests changing this, I will have to discuss this with him.
    if max_u_ind > N-1:
        max_u_ind = N-1
        kernel = kernel[:,0:max_u_ind-min_u_ind]
    
    if max_v_ind > N-1:
        max_v_ind = N-1
        kernel = kernel[0:max_v_ind-min_v_ind,:]

    if min_u_ind < 0:
        min_u_ind = 0
        kernel = kernel[:,min_u_ind:max_u_ind]

    if min_v_ind < 0:
        min_v_ind = 0
        kernel = kernel[min_v_ind:max_v_ind,:]

    array_subsec = uv_array[min_v_ind:max_v_ind, min_u_ind:max_u_ind]

    try:
        array_subsec += kernel
    except ValueError:
        print('Value Error')
        print('kernel shape {0}'.format(kernel.shape))
        print('kernel width u = %4i, kernel width v = %4i' % (width_u,width_v))
        print('Kernel shape (%4i,%4i)' % (max_v_ind-min_v_ind,max_u_ind-min_u_ind))
        print('Array size = %4i, u indexing size = %4i' % (len(uv_array), u_ind + width_u +1))
        print('Array size = %4i, v indexing size = %4i' % (len(uv_array), u_ind + width_u +1))

def Plot_Power_spec1D(Vis_power1D_list,radius,label_list=None,xlim=None,ylim=None,**kwargs):

    # Vis_power1D can be a list of multiple 1D power spectrums.
    # label_list should contain the same number of elements as Vis_power1D_list

    #print(np.shape(Vis_power1D_list))
    #print(label_list)

    # Initialising the figure object.
    # Need fig object, code breaks otherwise, fix this in the future.
    fig, axs = plt.subplots(1, figsize = (14,12), dpi=75)
    
    plt.semilogy()

    if len(np.shape(Vis_power1D_list)) < 2:
        axs.plot(radius,Vis_power1D_list,**kwargs)

    # Plotting multiple 1D power spectra if required.    
    elif label_list != None and len(np.shape(Vis_power1D_list)) > 1:
        for i in range(len(Vis_power1D_list)):
            axs.plot(radius,Vis_power1D_list[i],label = label_list[i],**kwargs)

    if xlim != None:
        axs.set_xlim(xlim)
    if ylim != None:
        axs.set_ylim(ylim)
    
    axs.set_xlabel(r'$\sqrt{u^2 + v^2}$',fontsize=24)
    axs.set_ylabel(r'$\rm{Power}$',fontsize=24)

    plt.legend(fontsize=24)

def Power_spec1D(Vis_power,u_arr,v_arr,r_vec=None):
    
    # Condition for radius vector. The default sampling is to just use
    # the (u,v) grid. If the user inputs a u and v vector this is specifically
    # for specifying their own sampling.    
    if np.any(r_vec) == None:
        u_vec = u_arr[0,:]
        v_vec = v_arr[:,0]
        
        # This is for binning the radii.
        r_vec = np.sqrt(u_vec[u_vec >= 0.0]**2 + v_vec[v_vec >= 0.0]**2)
    else:
        pass

    # The u_arr and v_arr should be shifted. 
    r_uv = np.sqrt(u_arr**2 + v_arr**2) + 0.00001

    # Initialising Power vector and Radius vector.
    Power_spec_1D = np.zeros(len(r_vec))
    Radius = np.zeros(len(r_vec))
    for i in range(len(r_vec)-1):
    
        Radius[i] = ((r_vec[i+1] + r_vec[i])/2.0)
        Power_spec_1D[i] = np.mean(Vis_power[np.logical_and(r_uv >= r_vec[i], r_uv <= r_vec[i+1])])# Weight.
        #print("#%3d, log_r[i] = %3.2f,log_r[i+1] = %3.2f, Power = %3.3e" % \
        #          (i,r_bins[i],r_bins[i+1],Power_spec_1D[i])) 

    Radius = np.roll(Radius,1)

    return Power_spec_1D, Radius

def realign_polar_xticks(ax):
    for x, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        if np.sin(x) > 0.1:
            label.set_horizontalalignment('right')
        if np.sin(x) < -0.1:
            label.set_horizontalalignment('left')

def Plot_img(Img,X_vec=None,Y_vec=None,projection=None,cmap='jet',figsize = (14,12),xlab=r'$l$',ylab=r'$m$',clab='Intensity',**kwargs):

    if projection:
        fig, axs = plt.subplots(1, figsize = figsize, dpi=75)

        # Creating the image objects:
        
        if np.any(X_vec) != None and np.any(Y_vec) != None:
            im = axs.imshow(Img,cmap=cmap,origin='upper',\
                                 extent=[np.min(X_vec),np.max(X_vec),np.min(Y_vec),np.max(Y_vec)])
            
        else:
            im = axs.imshow(Img,cmap=cmap,origin='upper')
            

        # Setting the colour bars:
        cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04)
        cb.set_label(label=clab)
    
        axs.set_xlabel(xlab)
        axs.set_ylabel(ylab)
    
        im.set_clim(**kwargs)

        plt.show()

    elif projection == "polar":
        
        fig = plt.figure(figsize = (14,12), dpi = 75)
        
        #label_size = 24
        font_size = 22
        #thetaticks = np.arange(0,360,45)
        
        ax1 = fig.add_subplot(111,projection='polar')
        pcm1 = ax1.pcolormesh(X_vec,Y_vec,Img, cmap = cmap)
        
        ax1.set_yticks([])
        ax1.set_theta_offset(np.pi/2.0)
        
        
        cb = fig.colorbar(pcm1, ax = ax1, fraction = 0.046, pad = 0.065)
    
        cb.set_label(label = 'Intensity', fontsize = font_size)
        cb.ax.tick_params(axis = 'x', labelsize = font_size - 2)
        
        realign_polar_xticks(ax1)
        
        plt.subplots_adjust(left=-0.5)
    
        pcm1.set_clim(**kwargs)
    
        plt.show()


def grid_old(grid_arr, u_coords, v_coords, vis, u_arr, v_arr, weighting='natural'):
    '''
    Natural and Gaussian kernel gridder. Will generalise in future.

        Parameters
        ----------
        grid_arr : numpy array, float
            Empty grid array.
        u_coords : numpy array, float
            1D array of visibilities u coordinates.
        v_coords : numpy array, float
            1D array of visibilities v coordinates.
        vis : numpy array, float
            1D array of complex visibilities.
        u_arr : numpy array, float
            2D Visibilities u array.
        v_arr : numpy array, float
            2D Visibilities u array.
        weighting : , string
            Default mode is natural, can also select a Gaussian Kernel.

        Returns
        -------
        2D Gridded visibility array and the corresponding weights array.
    '''

    start_temp = time.perf_counter()

    # Weight array, we will divide the entire container array by this.
    weights_arr = np.zeros(np.shape(grid_arr),dtype=complex)

    # Resolution required for both weighting cases.
    delta_u = np.abs(u_arr[0,1] - u_arr[0,0])
    delta_v = np.abs(v_arr[1,0] - v_arr[0,0])

    if weighting == 'natural':
        # Default case.
        u_vec = u_arr[0,:] 
        v_vec = v_arr[:,0]

    elif weighting == 'gaussian':

        # Initialiasing Gaussian kernel size.
        sig_x = 2 # Size in pixel space. 
        sig_y = 2
        
        # Converting to u,v coordinates.
        sig_u = sig_x * delta_u # Size in u,v space.
        sig_v = sig_y * delta_v
    else:
        pass

    # Looping through each visibility.
    for i in np.arange(len(vis)):
        
        if weighting == 'natural':

            # Determining the index location for each visibility.
            u_cent_ind = np.isclose(u_vec,u_coords[i],atol=delta_u/2.0)
            v_cent_ind = np.isclose(v_vec,v_coords[i],atol=delta_v/2.0)

            weights_arr[u_cent_ind,v_cent_ind] = weights_arr[u_cent_ind,v_cent_ind] + np.complex(1,0)

            grid_arr[u_cent_ind,v_cent_ind] = grid_arr[u_cent_ind,v_cent_ind] + vis[i]


        elif weighting == 'gaussian':

            temp_gauss_weights = gaussian(u_arr, v_arr, sig_u, sig_v, u_coords[i], v_coords[i])

            weights_arr =+ temp_gauss_weights # Adding Gaussian weights to weights arr.

            grid_arr =+ vis[i]*temp_gauss_weights # Adding gridded visibilities.

    
    end_temp = time.perf_counter()
    print('Grid time = %6.3f s' %  (end_temp - start_temp))

    if weighting == 'gaussian':
        # Checking that the weights sum to one. They should be normalised.
        print(np.sum(temp_gauss_weights))

    elif weighting == 'natural':
        print(np.sum(weights_arr),np.sum(grid_arr))

    # The weight array should always be positive in this context. 
    grid_arr[weights_arr > 0.0] /= weights_arr[weights_arr > 0.0]

    return grid_arr, weights_arr

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
            #vis_sky_deg[i] = np.sum(vis_sub*vis_ker)/np.sum(vis_ker)
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
        u_ind,v_ind = find_closet_xy(u=u,v=v,u_range=u_range,v_range=v_range)

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
            #weights_arr += kernel_array
            weights_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind] += kernel_array
            
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

"""
# Beam visibility code. This will need to be revisited in the future.
def Vis_Beam_Poly2D(U,V,dL,dM,l0,m0,Az0,Zen0,*a):
    
    a = np.array(a).ravel() # Setting the beam parameters.
    
    vis = np.zeros(np.shape(U),dtype=complex) # Initialising the vis array.
    p = int(np.sqrt(len(a)) - 1) # Forcing type.
    
    # Shifting the U and V arrays.
    U = fftshift(U)
    V = fftshift(V)
    
    Az0 = np.radians(Az0)
    Zen0 = np.radians(Zen0)
    
    #dL = dL*np.sqrt((np.sin(PA))**2 + (np.cos(PA)*np.cos(Zen0))**2)
    #dM = dM*np.sqrt((np.cos(PA))**2 + (np.sin(PA)*np.cos(Zen0))**2)
    
    print(dL,dM)
    
    #dL = dL*np.sqrt((np.sin(Az0))**2 + (np.cos(Az0)*np.cos(Zen0))**2)
    #dM = dM*np.sqrt((np.cos(Az0))**2 + (np.sin(Az0)*np.cos(Zen0))**2)
    
    print(dL,dM)
    
    index = 0
    for r in range(p+1):
        for s in range(p+1):
            
            # u-component:
            FT_b_u = 0
            for n in range(p-r+1):
                temp_frac_u = ((-1)**n) * ( ((dL/2.0)**(p-r-n)) / ((2*np.pi*(U))**(n+1)) )
                temp_cos_u = np.cos((np.pi/2.0)*(3*(p-r-n) + 1) - np.pi*(U)*dL)
                
                temp_u = temp_frac_u*temp_cos_u
                
                FT_b_u = FT_b_u + temp_u
                    
            ### There is an issue here. The limits are correct. But there is an issue
            ### with the order in which they are assigned depending on the polynomial order.

            
            #if p-r == 0:
            #    FT_b_u[np.isinf(FT_b_u)] = dL/2
                
            
            # Taking care of the discontinuities.
            if n==0:
            #if r==0:
                # Sinc function condition.
                #FT_b_u[np.isinf(FT_b_u)] = np.max(FT_b_u[np.isinf(FT_b_u)==False])
                FT_b_u[np.isinf(FT_b_u)] = dL/2
            
            if n==1:
            #if r==1:
                FT_b_u[np.isnan(FT_b_u)] = 0.0
            
            if n==2:
            #if r==2:
                #FT_b_u[np.isnan(FT_b_u)] = np.min(FT_b_u[np.isnan(FT_b_u)==False])
                FT_b_u[np.isnan(FT_b_u)] = -(dL**3)/12.0
                #FT_b_u[np.isnan(FT_b_u)] = -((dL)**3)/6
                    
            cond_u = True
            if r == 0 and cond_u == True:
                
                print(np.max(U[0,:]),np.min(U[0,:]),FT_b_u[0,:][U[0,:]==0.0])
                print(dL)
                plt.clf()
                #plt.semilogy(U[0,:],FT_b_u[0,:])
                plt.plot(U[0,:],FT_b_u[0,:])
                plt.xlabel(r'$u$',fontsize=24)
                plt.ylabel(r'$\frac{\hat{b}^2_2(u)}{2i^{2}e^{-2\pi i u l_0}}$',fontsize=24)
                plt.xlim([-25,25])
                #plt.plot(U[0,:],temp_cos_u[0,:])
                #plt.savefig('{0}.png'.format(n))
                plt.show()
                
            # v-component:
            FT_b_v = 0
            for q in range(s+1):
                temp_frac_v = ((-1)**q)*(((dM/2.0)**(s-q))/(2*np.pi*(V))**(q+1))
                temp_cos_v = np.cos((np.pi/2.0)*(3*(s-q) + 1) - np.pi*(V)*dM)
                
                temp_v = temp_frac_v*temp_cos_v
                
                FT_b_v = FT_b_v + temp_v
                
                if s==0:
                    # Sinc function condition.
                    FT_b_v[np.isinf(FT_b_v)] = dM/2
                
                if s==1:
                    FT_b_v[np.isnan(FT_b_v)] = 0.0
                
                if s==2:
                    #FT_b_v[np.isnan(FT_b_v)] = np.min(FT_b_v[np.isnan(FT_b_v)==False])
                    FT_b_v[np.isnan(FT_b_v)] = -(dM**3)/12.0
                
            cond_v = False
            if s == 2 and cond_v == True:
                
                print(np.max(V[:,0]),np.min(V[:,0]),FT_b_v[:,0][V[:,0]==0.0])
                print(dM)
                plt.clf()
                #plt.semilogy(U[0,:],FT_b_u[0,:])
                plt.plot(V[:,0],FT_b_v[:,0])
                #plt.savefig('{0}.png'.format(n))
                plt.show()
                

            
            vis = vis + 4*(complex(0,1)**(p-r-s))*a[index]*FT_b_u*FT_b_v
            
            index = index + 1
            
    # Exponential phase offset term.
    phase_l0m0 = np.zeros(np.shape(U),dtype=complex)
    #phase_l0m0.real = np.cos(-2*np.pi*(l0*U + m0*V))
    phase_l0m0.real = np.cos(-2*np.pi*(m0*U + l0*V))
    #phase_l0m0.imag = np.sin(-2*np.pi*(l0*U + m0*V))
    phase_l0m0.imag = np.sin(-2*np.pi*(m0*U + l0*V))
    
    # shifting the phase for the off centre window.
    vis = vis*phase_l0m0
    return vis

"""