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