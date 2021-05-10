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

sys.path.append(os.path.abspath("/home/jaiden/Documents/EoR/OSIRIS"))
import Iris


def find_closet_uv(u,v,u_vec,v_vec):
    '''
    Finds the indices for the (u,v) point associated to a u, v grid.

        Parameters
        ----------
        u : numpy array, float
            Baseline u value.
        v : numpy array, float
            Baseline v value.
        u_vec : numpy array, float
            Regular 1D u grid.
        v_vec : numpy array, float
            Regular 1D u grid.

        Returns
        -------
        Returns closest (u,v) indices.

    Author : J. Line
    '''
    u_resolution = np.abs(u_vec[1] - u_vec[0])
    v_resolution = np.abs(v_vec[1] - v_vec[0])
    
    ##Find the difference between the gridded u coords and the desired u
    u_offs = np.abs(u_vec - u)

    ##Find out where in the gridded u coords the current u lives;
    ##This is a boolean array of length len(u_offs)
    u_true = u_offs < u_resolution/2.0
    
    ##Find the index so we can access the correct entry in the container
    u_ind = np.where(u_true == True)[0]

    ##Use the numpy abs because it's faster (np_abs)
    v_offs = np.abs(v_vec - v)
    v_true = v_offs < v_resolution/2.0
    v_ind = np.where(v_true == True)[0]

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

    u_offs = u_vec - u
    v_offs = v_vec - v

    #u_off = -(u_offs[u_ind] / u_resolution)
    #v_off = -(v_offs[v_ind] / v_resolution)

    return u_ind,v_ind


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

    return gaussian

def Vis_degrid(u_arr,v_arr,u_vec,v_vec,u,v,vis_true,kernel_size=31, sig_u=1, sig_v=1):
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
        temp_u_ind, temp_v_ind = Iris.find_closet_xy(u[i],v[i],u_vec,v_vec)

        # Determining the index ranges:
        min_u_ind = temp_u_ind - int(kernel_size/2)
        max_u_ind = temp_u_ind + int(kernel_size/2) + 1
        min_v_ind = temp_v_ind - int(kernel_size/2)
        max_v_ind = temp_v_ind + int(kernel_size/2) + 1

        # Creating temporary u and v arrays.
        u_temp_arr = u_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind]
        v_temp_arr = v_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind]

        temp_gauss_weights = gaussian_kernel(u_temp_arr, v_temp_arr, sig_u, sig_v, u[i], v[i])

        # Might have to define a visibility subset that is larger.
        # Defining the visibility subset:
        vis_sub = vis_true[min_v_ind:max_v_ind, min_u_ind:max_u_ind]
        
        # Weighted average degridded visibilitiy.
        vis_deg[i] = np.sum(vis_sub*temp_gauss_weights)/np.sum(temp_gauss_weights)

    return vis_deg