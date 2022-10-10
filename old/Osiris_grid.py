#!/usr/bin/python

__author__ = "Jaiden Cook, Jack Line"
__credits__ = ["Jaiden Cook","Jack Line"]
__version__ = "0.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"

from cgi import test
from cmath import nan
from operator import mod
import numpy as np

import Osiris

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
        u_cent_ind,v_cent_ind = Osiris.find_closest_xy(u_coords[i],v_coords[i],u_vec,v_vec)

        weights_arr[v_cent_ind,u_cent_ind] = weights_arr[v_cent_ind,u_cent_ind] + 1
        grid_arr[v_cent_ind,u_cent_ind] = grid_arr[v_cent_ind,u_cent_ind] + vis[i]

    # The weight array should always be positive in this context. 
    grid_arr[weights_arr > 0.0] /= weights_arr[weights_arr > 0.0]

    #grid_arr = grid_arr/np.sum(weights_arr)

    #end_temp = time.perf_counter()
    #print('Grid time = %6.3f s' %  (end_temp - start_temp))

    return grid_arr, weights_arr


def grid_gaussian(grid_arr, u_coords, v_coords, vis, u_grid, v_grid, \
    u_vec, v_vec, kernel_size=51, sig_grid=2.16):
    '''
    Natural and Gaussian kernel gridder. Will generalise in future.

        Parameters
        ----------
        grid_arr : numpy array, float
            Empty 2D complex grid array.
        u_coords : numpy array, float
            1D array of visibilities u coordinates.
        v_coords : numpy array, float
            1D array of visibilities v coordinates.
        vis : numpy array, float
            1D array of complex visibilities.
        u_grid : numpy array, float
            2D Visibilities u grid.
        v_grid : numpy array, float
            2D Visibilities u grid.

        Returns
        -------
        2D Gridded visibility array and the corresponding weights array.
    '''
    # Weight array, we will divide the entire container array by this.
    weights_arr = np.zeros(np.shape(grid_arr))
    Nvis_arr = np.zeros(np.shape(grid_arr))

    # Resolution required for both weighting cases.
    delta_u = np.abs(u_grid[0,1] - u_grid[0,0])
    #delta_v = np.abs(v_grid[1,0] - v_grid[0,0])

    # sig_grid should already be in wavelengths.

    # Optionally here we could specify sig_grid in terms of lambda, the conversion would be
    # sig_grid_pix = sig_grid_lam / du.
    sig_u = sig_grid #* delta_u, size in uv space pixels.
    sig_v = sig_grid #* delta_v, size in uv space pixels.

    # Turn this into a real condition if this works for mult time steps.
    test_cond = False
    if test_cond:
        # Testing a slightly adjusted visibility weighting scheme.
        # origin set to zero.
        u_vec_O = (np.arange(kernel_size) - (kernel_size-1)/2)*delta_u
        v_vec_O = (np.arange(kernel_size) - (kernel_size-1)/2)*delta_u

        u_temp_arr, v_temp_arr = np.meshgrid(u_vec_O,v_vec_O)

        temp_gauss_weights = Osiris.gaussian_kernel(u_temp_arr, v_temp_arr, sig_u, sig_v, 0, 0)
    else:
        pass

    blackman_wind_cond = False
    if blackman_wind_cond:
        FWHM = 2*np.sqrt(2*np.log(2))*sig_u

        #Nside = int(np.ceil(FWHM)/delta_u +1) # 13 if FWHM/du = 12.
        Nside = int(np.floor(FWHM/0.3432)/delta_u +1) # 11 if FWHM/du = 12.

        from scipy import signal

        window = signal.blackmanharris(int(Nside))
        #Normalise this.
        temp_gauss_weights = np.sqrt(np.outer(window,window))
        temp_gauss_weights = 4*temp_gauss_weights/np.sum(temp_gauss_weights)

        kernel_size = Nside
    else:
        pass


    # Looping through each visibility.
    for i in np.arange(len(vis)):
        
        # Determining the index location for each visibility.
        u_cent_ind,v_cent_ind = Osiris.find_closest_xy(u_coords[i],v_coords[i],u_vec,v_vec)

        # Determining the index ranges:
        min_u_ind = u_cent_ind - int(kernel_size/2)
        max_u_ind = u_cent_ind + int(kernel_size/2) + 1
        min_v_ind = v_cent_ind - int(kernel_size/2)
        max_v_ind = v_cent_ind + int(kernel_size/2) + 1

        if test_cond:
            pass
        else:
            # Creating temporary u and v arrays.
            u_temp_arr = u_grid[min_v_ind:max_v_ind, min_u_ind:max_u_ind]
            v_temp_arr = v_grid[min_v_ind:max_v_ind, min_u_ind:max_u_ind]

            temp_gauss_weights = Osiris.gaussian_kernel(u_temp_arr, v_temp_arr, sig_u, sig_v, u_coords[i], v_coords[i])


        # Adding Gaussian weights to weights arr.
        weights_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind] = \
            weights_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind] + temp_gauss_weights

        # Adding Gaussian weights to weights arr.
        #Nvis_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind] = \
        #    Nvis_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind] + 1 

        # Adding gridded visibilities.
        grid_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind] = \
            grid_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind] + vis[i]*temp_gauss_weights

        # Adding gridded visibilities.
        #grid_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind] = \
        #    grid_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind] + temp_gauss_weights*(1 + 1j)

    
    #### uncomment 14/9/22 ###
    grid_arr[weights_arr > 0.0] = grid_arr[weights_arr > 0.0]/weights_arr[weights_arr > 0.0]


    if sig_u < delta_u:
        # Safeguard against nan, and ing value swhen sig < du.
        grid_arr.real[np.isinf(grid_arr.real)] = 0
        grid_arr.real[np.isnan(grid_arr.real)] = 0
        grid_arr.imag[np.isnan(grid_arr.imag)] = 0
        grid_arr.imag[np.isinf(grid_arr.imag)] = 0
    
    # For multiple time steps weights need to be normalised. This may not be the best approach.
    # This is similar to calculating the average weight. Probably not the best approach.
    
    test_cond = False
    if test_cond:
        # Testing the grid kernel size.
        name = 'gridding-kernel-sig-{0}-v2'.format(np.round(sig_grid,2))
        out_path = '/home/jaiden/Documents/Skewspec/output/'
        np.savez_compressed(out_path + name, grid_arr = temp_gauss_weights, u_temp_arr = u_temp_arr, \
            v_temp_arr = v_temp_arr, u = u_coords[i], v = v_coords[i])
    else:
        pass

    return grid_arr, weights_arr


def grid_cube(u_coords_arr,v_coords_arr,vis_arr,u_grid,v_grid,\
    grid_arr_cube,vis_weights_cube,weighting='gaussian',kernel_size=91,sig_grid=4):
    '''
    Wrapper function for iteratively gridding visibility frequency slices.

        Parameters
        ----------
        u_coords_arr : numpy array, float
            2D array of u [lambda] baseline values Nbaselines x Nchans.
        v_coords_arr : numpy array, float
            2D array of 2 [lambda] baseline values Nbaselines x Nchans.
        vis_arr : numpy array, float
            Complex 2D array of visibilities Nbaselines x Nchans.
        u_arr : numpy array, float
            2D Visibilities u grid.
        v_arr : numpy array, float
            2D Visibilities u grid.
        grid_arr_cube : numpy array, float
            Container for gridded visibilities.
        vis_weights_cube : numpy array, float
            Containter for gridded weights.
        weighting : string,
            Specify the type of gridding 'natural' or 'gaussian' which is default.
        kernel : integer,
            Grid Gaussian kernel pixel size, default is 7.
        sig_grid : float,
            Width of Gaussian gridding kernel, default size is 0.5 wavelengths. 

        Returns
        -------
        grid_arr_cube : numpy array, float
            3D complex visibilities grid cube.
        vis_weights_cube : numpy array, float
            3D cube of gridded weightes. 
    '''
    # Number of iterations.
    #N_iter = len(u_coords_list)
    N_iter = u_coords_arr.shape[1]
    u_vec = u_grid[0,:]
    v_vec = v_grid[:,0]

    import sys
    test_cond = False

    for i in range(N_iter):
        #Looping through each frequency channel.
        Nvis_tmp = len(u_coords_arr[:,i][np.abs(u_coords_arr[:,i])>0])

        # Progress bar:
        if (i+1) % 10 == 0:
            sys.stdout.write("\rChannels processed: {0}/{1}".format((i+1),N_iter))
            sys.stdout.flush()
        elif (i+1) == N_iter:
            # Last iteration.
            sys.stdout.write("\rChannels processed: {0}/{0}\n".format(N_iter))
            sys.stdout.flush()
        else:
            pass

        u_coords = u_coords_arr[:Nvis_tmp,i]
        v_coords = v_coords_arr[:Nvis_tmp,i]
        vis_vec = vis_arr[:Nvis_tmp,i]

        if weighting == 'natural':
        # Default case.
        
            #grid_natural(grid_arr, u_coords, v_coords, vis, u_vec, v_vec)
            grid_arr_cube[:,:,i],vis_weights_cube[:,:,i] = grid_natural(grid_arr_cube[:,:,i], \
                u_coords, v_coords, vis_vec, u_vec, v_vec)

        elif weighting == 'gaussian':

            #grid_gaussian(grid_arr, u_coords, v_coords, vis, u_grid, v_grid, u_vec, v_vec)
            grid_arr_cube[:,:,i],vis_weights_cube[:,:,i] = grid_gaussian(grid_arr_cube[:,:,i], \
                u_coords, v_coords, vis_vec, u_grid, v_grid, u_vec, v_vec, \
                kernel_size=kernel_size, sig_grid=sig_grid)
            
            if i == 0 and test_cond:
                # Testing the weight outputs.
                name = 'Gauss-slice-ntime_step-test-nt8'
                out_path = '/home/jaiden/Documents/Skewspec/output/'
                np.savez_compressed(out_path + name, grid_slice = grid_arr_cube[:,:,i],
                    weights_slice=vis_weights_cube[:,:,i])
                print(np.sum(grid_arr_cube[:,:,i].real),np.sum(grid_arr_cube[:,:,i].imag))


    return grid_arr_cube, vis_weights_cube
