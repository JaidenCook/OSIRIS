#!/usr/bin/python

__author__ = "Jaiden Cook, Jack Line"
__credits__ = ["Jaiden Cook","Jack Line"]
__version__ = "0.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"

from numba import jit
import numpy as np
import time

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


def grid_natural(grid_arr, u_coords, v_coords, vis, u_vec, v_vec):
    '''
    Natural weighting visibility kernel gridder.

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
        u_vec : numpy array, float
            1D Visibilities u array.
        v_vec : numpy array, float
            1D Visibilities u array.
        
        Returns
        -------
        2D Gridded visibility array and the corresponding weights array.
    '''
    #start_temp = time.perf_counter()

    # Weight array, we will divide the entire container array by this.
    weights_arr = np.zeros(np.shape(grid_arr))

    # Looping through each visibility.
    for i in np.arange(len(vis)):
        
        # Determining the index location for each visibility.
        u_cent_ind,v_cent_ind = Iris.find_closet_xy(u_coords[i],v_coords[i],u_vec,v_vec)

        weights_arr[u_cent_ind,v_cent_ind] = weights_arr[u_cent_ind,v_cent_ind] + 1
        grid_arr[u_cent_ind,v_cent_ind] = grid_arr[u_cent_ind,v_cent_ind] + vis[i]
    
    #print(np.sum(weights_arr),np.sum(grid_arr))

    # The weight array should always be positive in this context. 
    grid_arr[weights_arr > 0.0] /= weights_arr[weights_arr > 0.0]

    #end_temp = time.perf_counter()
    #print('Grid time = %6.3f s' %  (end_temp - start_temp))

    return grid_arr, weights_arr


def grid_gaussian(grid_arr, u_coords, v_coords, vis, u_arr, v_arr, kernel_size=31, sig_x=2, sig_y=2):
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

        Returns
        -------
        2D Gridded visibility array and the corresponding weights array.
    '''
    # Weight array, we will divide the entire container array by this.
    weights_arr = np.zeros(np.shape(grid_arr))

    # Resolution required for both weighting cases.
    delta_u = np.abs(u_arr[0,1] - u_arr[0,0])
    delta_v = np.abs(v_arr[1,0] - v_arr[0,0])

    # Converting to u,v coordinates.
    sig_u = sig_x * delta_u # Size in u,v space.
    sig_v = sig_y * delta_v

    # Default case.
    u_vec = u_arr[0,:] 
    v_vec = v_arr[:,0]

    # Looping through each visibility.
    for i in np.arange(len(vis)):
        
        # Determining the index location for each visibility.
        u_cent_ind,v_cent_ind = Iris.find_closet_xy(u_coords[i],v_coords[i],u_vec,v_vec)

        # Determining the index ranges:
        min_u_ind = u_cent_ind - int(kernel_size/2)
        max_u_ind = u_cent_ind + int(kernel_size/2) + 1
        min_v_ind = v_cent_ind - int(kernel_size/2)
        max_v_ind = v_cent_ind + int(kernel_size/2) + 1

        # Creating temporary u and v arrays.
        u_temp_arr = u_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind]
        v_temp_arr = v_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind]

        temp_gauss_weights = gaussian_kernel(u_temp_arr, v_temp_arr, sig_u, sig_v, u_coords[i], v_coords[i])

        # Adding Gaussian weights to weights arr.
        weights_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind] = \
            weights_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind] + temp_gauss_weights 
        # Adding gridded visibilities.
        grid_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind] = \
            grid_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind] + vis[i]*temp_gauss_weights 

    # The weight array should always be positive in this context. 
    grid_arr[weights_arr > 0.0] /= weights_arr[weights_arr > 0.0]

    return grid_arr, weights_arr


def grid_cube(u_coords_list,v_coords_list,vis_list,u_arr,v_arr,\
    grid_arr_cube,vis_weights_cube,weighting='natural'):
    '''
    Wrapper function for iteratively gridding visibility frequency slices.

        Parameters
        ----------
        u_coords_list : numpy array, float
            Empty grid array.
        v_coords_list : numpy array, float
            1D array of visibilities u coordinates.
        vis_list : numpy array, float
            1D array of visibilities v coordinates.
        u_arr : numpy array, float
            2D Visibilities u array.
        v_arr : numpy array, float
            2D Visibilities u array.
        grid_arr_cube : numpy array, float
            Container for gridded visibilities.
        vis_weights_cube : numpy array, float
            Containter for gridded weights.
        weighting : string,
            Specify the type of gridding 'natural' which is default or 'gaussian'.

        Returns
        -------
        3D gridded visibility cube and weights cube.
    '''
    # Number of iterations.
    N_iter = len(u_coords_list)

    if weighting == 'natural':

        # Default case.
        u_vec = u_arr[0,:] 
        v_vec = v_arr[:,0]

        for i in range(N_iter):

            #grid_natural(grid_arr, u_coords, v_coords, vis, u_vec, v_vec)
            grid_arr_cube[:,:,i],vis_weights_cube[:,:,i] = grid_natural(grid_arr_cube[:,:,i], \
                u_coords_list[i], v_coords_list[i], vis_list[i], u_vec, v_vec)

    elif weighting == 'gaussian':

        for i in range(N_iter):

            #grid_gaussian(grid_arr, u_coords, v_coords, vis, u_arr, v_arr)
            grid_arr_cube[:,:,i],vis_weights_cube[:,:,i] = grid_gaussian(grid_arr_cube[:,:,i], \
                u_coords_list[i], v_coords_list[i], vis_list[i], u_arr, v_arr)

    return grid_arr_cube, vis_weights_cube
