#!/usr/bin/python

__author__ = "Jaiden Cook, Jack Line"
__credits__ = ["Jaiden Cook","Jack Line"]
__version__ = "0.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"

import numpy as np
import time
import sys
import os

from numpy.lib.index_tricks import nd_grid

import Iris

# Inserted text is a remote test.

def Vis_degrid_gaussian(u_arr,v_arr,u_vec,v_vec,u,v,vis_true,kernel_size=31, sig_u=2, sig_v=2):
    """
    Visibility degridding function. Uses an input kernel, and uv point list to degrid
    visibilities.

    Parameters
        ----------
        u_arr : numpy array, float
            2D Visibilities u array.
        v_arr : numpy array, float
            2D Visibilities u array.
        u_vec : numpy array, float
            1D Visibilities u array.
        v_vec : numpy array, float
            1D Visibilities u array.
        u : numpy array, float
            1D array of visibilities u coordinates.
        v : numpy array, float
            1D array of visibilities v coordinates.
        vis_true : numpy array, float
            2D array of complex visibilities.
        

        Returns
        -------
        Weighted average of visibilities, corresponding to (u,v) points.
    """
    # Initialising the new deridded visibility array:
    vis_deg = np.zeros(len(u),dtype=complex)
    
    for i in range(len(u)):
    
        # These should be the indices of the coordinates closest to the baseline. These coordinates
        # should line up with the kernel.
        temp_u_ind, temp_v_ind = Iris.find_closest_xy(u[i],v[i],u_vec,v_vec)

        # Determining the index ranges:
        min_u_ind = temp_u_ind - int(kernel_size/2)
        max_u_ind = temp_u_ind + int(kernel_size/2) + 1
        min_v_ind = temp_v_ind - int(kernel_size/2)
        max_v_ind = temp_v_ind + int(kernel_size/2) + 1

        # Creating temporary u and v arrays.
        u_temp_arr = u_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind]
        v_temp_arr = v_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind]

        temp_gauss_weights = Iris.gaussian_kernel(u_temp_arr, v_temp_arr, sig_u, sig_v, u[i], v[i])

        # Might have to define a visibility subset that is larger.
        # Defining the visibility subset:
        vis_sub = vis_true[min_v_ind:max_v_ind, min_u_ind:max_u_ind]
        
        # Weighted average degridded visibilitiy.
        vis_deg[i] = np.sum(vis_sub*temp_gauss_weights)/np.sum(temp_gauss_weights)
        #vis_deg[i] = np.sum(vis_sub*temp_gauss_weights)#/np.sum(temp_gauss_weights)

    #print(np.sum(temp_gauss_weights))
    #vis_deg = vis_deg/len(vis_deg)

    return vis_deg

def Vis_degrid(kernel,u_vec,v_vec,u,v,vis_true,w=None,phase_cond=False):
    """
    Visibility degridding function. Uses an input kernel, and uv point list to degrid
    visibilities.

    Parameters
        ----------
        kernel : object, float
            Kernel object, contains 2D kernel arry which can be complex.
        u_vec : numpy array, float
            1D Visibilities u array.
        v_vec : numpy array, float
            1D Visibilities u array.
        u : numpy array, float
            1D array of visibilities u coordinates.
        v : numpy array, float
            1D array of visibilities v coordinates.
        vis_true : numpy array, float
            2D array of complex visibilities.
        w : numpy array, float
            Default value is None. If None no w-projection is performed. If a w_vec is given
            w-projection is performed. Must be the same shape as u and v.
        

        Returns
        -------
        Weighted average of visibilities, corresponding to (u,v) points.
    """

    # Need to change some parameters here. 
    kernel_size = len(kernel.kernel)

    # Initialising the new deridded visibility array:
    vis_deg = np.zeros(len(u),dtype=complex)

    # Setting the kernel to w=0, this is the default for no w-projection. If w != None then
    # the w-kernel is calculated and overwritten in the for loop below.
    kernel.calc_w_kernel(w=0.0)

    for i in range(len(u)):
    
        # These should be the indices of the coordinates closest to the baseline. These coordinates
        # should line up with the kernel.
        if phase_cond:
            # Condition if phase offset is true.
            u_ind, v_ind, u_off, v_off = Iris.find_closest_xy(u[i],v[i],u_vec,v_vec,off_cond=phase_cond)
        else:
            # Default condition don't return the offsets.
            u_ind, v_ind = Iris.find_closest_xy(u[i],v[i],u_vec,v_vec)
            u_off=0
            v_off=0

        # Determining the index ranges:
        min_u_ind = u_ind - int(kernel_size/2)
        max_u_ind = u_ind + int(kernel_size/2) + 1
        min_v_ind = v_ind - int(kernel_size/2)
        max_v_ind = v_ind + int(kernel_size/2) + 1

        # Might have to define a visibility subset that is larger.
        # Defining the visibility subset:
        vis_sub = vis_true[min_v_ind:max_v_ind, min_u_ind:max_u_ind]
        #vis_sub = vis_true[min_u_ind:max_u_ind, min_v_ind:max_v_ind]
        
        if np.any(w):
            # If vector of w values is given.
            # Calculating the w-kernel.
            #kernel.calc_w_kernel(w[i],u_off,v_off)
            kernel.calc_w_kernel(w[i],-u_off,-v_off)
        else:
            #kernel.calc_w_kernel(0.0,u_off,v_off)
            kernel.calc_w_kernel(0.0,-u_off,-v_off)

        # Weighted average degridded visibilitiy.
        vis_deg[i] = np.sum(vis_sub*kernel.w_kernel)#/np.sum(kernel.w_kernel)

        #if i==100:
        #    print(np.sum(kernel.w_kernel).real)
        #    kernel.plot_kernel(title='test_kernel')
        #    print('Abs sum')
        #    print(np.sum(np.abs(kernel.w_kernel)))
        #    print('real sum')
        #    print(np.sum(np.real(kernel.w_kernel)),np.sum(np.imag(kernel.w_kernel)))
            


    #print(np.sum(temp_gauss_weights))
    #vis_deg = vis_deg/len(vis_deg)

    return vis_deg


class w_kernel():
    """
    Creates the input arrays for calculating the w-sky-kernel, and saves them. Takes input w-terms
    and outputs the corresponding w-sky-kernel.

    ...

    Attributes
    ----------
    n_ker_arr : numpy array
    l_vec : numpy array
    m_vec : numpy array
        ...


    Methods
    -------
    calc_w_sky_kernel(w=""):
        ...
    """
    
    def __init__(self,beam_grid,l_grid,m_grid):
        # Default parameters are for an all-sky image. 
        
        # Initialising kernel vectors.
        self.beam_grid = beam_grid
        self.l_grid = l_grid
        self.m_grid = m_grid

        # Creating kernel radius.
        r_grid = np.sqrt(l_grid**2 + m_grid**2)

        # Initialising n_arr.
        self.n_grid = np.zeros(np.shape(r_grid))

        # Populating n_arr.
        self.n_grid[r_grid < 1] = np.sqrt(1 - l_grid[r_grid < 1]**2 - m_grid[r_grid < 1]**2)

        #beam_grid[r_grid < 1] = beam_grid[r_grid < 1]/self.n_grid[r_grid < 1]
        self.kernel = beam_grid

    def calc_w_kernel(self,w,u_off=None,v_off=None):

        from scipy.fft import ifftn,fftn,fftfreq,fftshift,ifftshift

        """
        Returns the w-sky-kernel.

        Parameters
        ----------
        w : float
            Baseline w-coordinate.
        
        Returns
        -------
        None
        """

        #print('w = %5.3f' % w)

        # Initialising constants:
        L = np.abs(np.max(self.l_grid) - np.min(self.l_grid))
        M = np.abs(np.max(self.m_grid) - np.min(self.m_grid))
        N = len(self.l_grid)

        # Calculating the w-sky-kernel:
        if np.any(u_off) and np.any(v_off):
            # Account for the offsets in relation to the (u,v) grid.
            offset_grid = np.exp(-2j*np.pi*(u_off*self.l_grid + v_off*self.m_grid))
            #offset_grid = np.exp(-2j*np.pi*(v_off*self.l_grid + u_off*self.m_grid))
            # Calculating.
            w_sky_ker = offset_grid*np.exp(-2j*np.pi*w*(self.n_grid - 1))*self.kernel
            #w_sky_ker = offset_grid*self.kernel
            #w_sky_ker = np.exp(-2j*np.pi*w*(self.n_grid - 1))*self.kernel
        else:
            # Default don't phase rotate relative to the offsets.
            w_sky_ker = np.exp(-2j*np.pi*w*(self.n_grid - 1))*self.kernel

        self.w_sky_ker = w_sky_ker

        # FFT w-sky-kernel:
        u_grid, v_grid, w_kernel = Iris.Visibilities_2D(w_sky_ker,L,M,N)

        # Setting attributes:
        self.u_grid = u_grid
        self.v_grid = v_grid

        # Setting and normalising the w-kernel.
        self.w_kernel = w_kernel/np.sum(np.abs(w_kernel))
        #self.w_kernel = w_kernel/np.abs(np.sum(w_kernel))

    def plot_kernel(self,ker='sky',real_cond=True,imag_cond=False,title=None,**kwargs):
        """
        Plotting method for the kernel. For diagnostic purposes.

        Parameters
        ----------
        sky : string
            String condition. Can either plot the sky kernel, or the w-kernel. Default is sky.
        real_cond : boolean
            Real array plotting condition. This is the default.
        imag_cond : boolean
            Imaginary arry plotting condition. Not the default, can plot both if both conditions
            are true.
        
        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt

        if ker == 'sky':
            if real_cond:
                # Plot real part of sky kernel:
                Iris.Plot_img(self.w_sky_ker.real,self.l_grid,self.m_grid,cmap='viridis',figsize=(5,5),\
                clab='Response',xlab=r'$l$',ylab=r'$m$',title=title,**kwargs)
            elif imag_cond:
                # Plot imag part of sky kernel:
                Iris.Plot_img(self.w_sky_ker.imag,self.l_grid,self.m_grid,cmap='viridis',figsize=(5,5),\
                clab='Response',xlab=r'$l$',ylab=r'$m$',title=title,**kwargs)
        elif ker == 'vis':
            if real_cond:
                # Plot real part of the w-kernel:
                Iris.Plot_img(self.w_kernel.real,self.u_grid,self.v_grid,cmap='viridis',figsize=(5,5),\
                clab='Response',xlab=r'$u\,[\lambda]$',ylab=r'$v\,[\lambda]$',title=title,**kwargs)
            elif imag_cond:
                # Plot imag part of the w-kernel:
                Iris.Plot_img(self.w_kernel.imag,self.u_grid,self.v_grid,cmap='viridis',figsize=(5,5),\
                clab='Response',xlab=r'$u\,[\lambda]$',ylab=r'$v\,[\lambda]$',title=title,**kwargs)

