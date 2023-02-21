#!/usr/bin/python

__author__ = "Jaiden Cook, Jack Line"
__credits__ = ["Jaiden Cook","Jack Line"]
__version__ = "1.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"


import numpy as np
import Osiris

def sigma_grid_calc(freq,D=4,epsilon=0.42,radian_cond=False):
    """
    Calculate the expected Gaussian grid kernel size. The beam
    can be approximated by sigma_b = (epsilon*c)/(freq*D). The Fourier grid
    kernel sigma_u = 1/(pi*sigma_b). Substituting in sigma_b yields sigma_u.
    We can use this grid size to calculate the Blackman-Harris window sizes.

        Parameters
        ----------
        freq : numpy array, float
            Vector of frequencies, or a single frequency value in Hz.
        D : float, default=4
            Diameter of the MWA tile in meters.
        epsilon : float, default=0.42
            Scaling factor from Airy disk to a Gaussian (Nasirudin et al. 2020).
        radian_cond : bool, default=False
            If true return the Fourier inverse sigma. That is the beam size in 
            direction cosines as projected onto the (l,m) plane. 
        
        Returns
        -------
        Sigma_u for a Gaussian kernel. Can return a single or numpy array.
    """
    
    c = 3e8 # m s^-1
    
    # Calculating the wavelength(s).
    lam = c/freq

    # The approximate beam size can be calculated by epsilon*lam/D 
    # or 1/(pi*sigma_u). 

    if radian_cond:
        # Direction cosine beam width.
        sigma = (epsilon*lam)/D
    else:
        # (u,v) grid beam width.
        sigma = D/(epsilon*np.pi*lam)

    return sigma

def gaussian_kernel(u_arr,v_arr,sig,du_vec,dv_vec,method=None,*args):
    """
    Generate A generic 2D Gassian kernel. For gridding and weighting purposes. If
    du_vec is a vector and not a float, then this function returns a cube of Gaussian
    weights, 

        Parameters
        ----------
        u_arr : numpy array, float
            2D Visibilities u array.
        v_arr : numpy array, float
            2D Visibilities v array.
        sig : numpy array, float
            Kernel size in wavelengths.
        du_vec : numpy array, float
            Visibility u coordinate centre, or difference between the u grid coord,
            and the visibility.
        dv_vec : numpy array, float
            Visibility v coordinate centre, or difference between the u grid coord,
            and the visibility.

        Returns
        -------
        2D Gaussian weights array.

    """

    try:
        # Case when there is more than one offset value du. du  is a vector.
        du_vec_shape = du_vec.shape

        u_bit = (u_arr[:,:,None] - du_vec[None,None,:])/sig
        v_bit = (v_arr[:,:,None] - dv_vec[None,None,:])/sig

        #u_bit = (u_arr[:,:,None] - du_vec[None,None,:])
        #v_bit = (v_arr[:,:,None] - dv_vec[None,None,:])
    except AttributeError:
        # Defualt case where this is only one offset.
        u_bit = (u_arr - du_vec)/sig
        v_bit = (v_arr - dv_vec)/sig

        #u_bit = (u_arr - du_vec)
        #v_bit = (v_arr - dv_vec)

    #amp = 1/(2*np.pi*sig**2)
    #gaussian = amp*np.exp(-0.5*(u_bit**2 + v_bit**2))

    # Fourier inverse of the sig=sig_u. This is the beam size.
    sig_beam = 1/(np.pi * sig)

    # Defined as the Fourier transform of a Gaussian fit to the Beam.
    amp = np.sqrt(np.pi)*sig_beam**2 # should be around 0.05
    gaussian = amp*np.exp(-(u_bit**2 + v_bit**2))

    # Note that sum(gaussian)*dA = 1 thus sum(gaussian) = 1/dA.
    # The integral of Gaussian is what is equal to 1. int ~ sum*dA

    # Setting thing greater than 2.5sig to zero.
    r_bit = np.sqrt(u_bit**2 + v_bit**2)
    gaussian[r_bit > 2.5*sig] = 0

    resolution = np.abs(u_arr[0,0]-u_arr[0,1])

    gaussian = (1/resolution**2)*gaussian/np.sum(gaussian,axis=(0,1))

    return gaussian


def blackman_harris2D(u_arr,v_arr,L,du_vec,dv_vec,method='radial'):
    """
    This function calculates the blackmann-harris kernel shifted relative to some
    dx value. We measure in x values, which corresponde to some grid, which should
    be centred at 0 for a a grid with an odd number of data points.

        Parameters
        ----------
        u_arr : numpy array, float
            Grid of x values.
        v_arr : numpy array, float
            Grid of yy values.
        L : numpy array, float
            Size of the kernel.
        du_vec : numpy array, float
            float or array of grid x offset values.
            grid kernel size
        dv_vec : numpy array, float
            float or array of grid y offset values.
            grid kernel size
        method : str
            Can be radial or square. 

        Returns
        -------
        kernel2D : float, array
            Array of shift values.
    """
    # Blackman Harris coefficeint values.
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168

    def blackman_harris1D(x,X,dx):

        X_shift = (x - dx + X*0.5)*2*np.pi/X
        window1D = a0 - a1*np.cos(X_shift) + a2*np.cos(2*X_shift) - a3*np.cos(3*X_shift)

        # Should be zero otherwise the kernel repeats.
        window1D[np.abs(x) > X/2] = 0

        return window1D

    resolution = np.abs(u_arr[0,0]-u_arr[0,1])

    try:
        shape_tup = du_vec.shape

        # In the case where there is more than one offset.
        r_arr_shift = \
            np.sqrt((u_arr[:,:,None] - du_vec[None,None,:])**2 + \
            (v_arr[:,:,None] - dv_vec[None,None,:])**2)

    except AttributeError:
        # Defualt case where this is only one offset.
        r_arr_shift = np.sqrt((u_arr-du_vec)**2 + (v_arr-dv_vec)**2)

        if method == 'square':
            # Stops the square method from breaking. Single value turned into 
            # numpy array. Makes it iterable.
            du_vec = np.array([du_vec])
            dv_vec = np.array([dv_vec])

    if method == 'radial':

        # Shifted x values.    
        r_prime = (r_arr_shift + L*0.5)*2*np.pi/L

        # Kernel.
        kernel2D = a0 - a1*np.cos(r_prime) + a2*np.cos(2*r_prime) - a3*np.cos(3*r_prime)
        
    elif method == 'square':
        # uvec and vvec
        u_vec = u_arr[0,:]
        v_vec = v_arr[:,0]
        
        # Radius factor attempts to account for the kernel size increase due to 
        # the square shape. Might adjust in future to match area of a Gaussian beam.
        radius_factor = 1
        window_vec_u_2D = np.array([blackman_harris1D(u_vec,L/radius_factor,dx) for dx in du_vec]).T
        window_vec_v_2D = np.array([blackman_harris1D(v_vec,L/radius_factor,dy) for dy in dv_vec]).T

        # Einstein notation used to perform outer product for 3 axes.
        kernel2D = np.einsum('j...,i...',window_vec_u_2D,window_vec_v_2D).T

        # Values outside the kernel should be set to 0.
    else:
        print('Input wrong method, default is "radial", alternative is "square".')
        return None

    # Values outside the kernel should be set to 0.
    dist_factor = 1/2
    kernel2D[r_arr_shift > dist_factor*L] = 0.0

    # Normalising so the weight integral is 1.
    kernel2D = (1/resolution**2)*kernel2D/np.sum(kernel2D,axis=(0,1))

    return kernel2D

def calc_weights_cube(u_shift_vec,v_shift_vec,du,
        sig=2.41,kernel_size=51,kernel='gaussian'):
    """
    This function calculates the weight kernel for each input visiblity.
    The output is a weights cube, where each slice is the gridding kernel
    for each visibility. This function also outputs the grid index centres
    for each visibility.

        Parameters
        ----------
        u_shift_vec : numpy array, float
            Vector of offset u-values.
        v_shift_vec : numpy array, float
            Vector of offset v-values.
        sig : numpy array, float
            Size of the kernel in wavelengths.
        du : numpy array, float
            Grid pixel size in wavelengths.
        kernel_size : numpy array, float
            grid kernel size in pixels.
        kernel : str
            Default is 'gaussian', options are 'blackman-harris', 'natural',
            and 'uniform'.


        Returns
        -------
        weights_cube : float, array
            Cube of weight kernels for each visibility.
        u_cent_ind_vec : int, array
            Integer array of u grid index centres.
        v_cent_ind_vec : int, array
            Integer array of v grid index centres.
    """

    if kernel == 'gaussian':
        # Gaussian weights cube function.
        func = gaussian_kernel # Assigning function namespace.

        size = sig
    elif kernel == 'blackman-harris':
        # Blackman-Harris weights cube function.
        func = blackman_harris2D # Assigning function namespace.

        sig_nu = sig/np.sqrt(2)

        # Gaussian FWHM, used to determine Blackman-Harris size.
        #FWHM = 2*np.sqrt(2*np.log(2))*sig
        FWHM = 2*np.sqrt(2*np.log(2))*sig_nu # For Gauss with width sig not root(2)sig.

        # L is the 1D length of the kernel.
        # 0.3432 is the ratio of the FWHM to the total Blackman-Harris
        # length. 
        # FWHM/size = 0.3432 for Blackman-Harris window.
        size = FWHM/0.3432

        # Kernel size is smaller for BH.
        kernel_size_nu = int(np.ceil(size)/du) # 11 if FWHM/du = 12.
        
        if (kernel_size_nu%2) == 0:
            kernel_size_nu += 1

        #print(kernel_size_nu)
        #print(f'Gaussian radius = {sig:5.3f} [lam]')
        #print(f'FWHM = {FWHM:5.3f} [lam]')

        if kernel_size_nu > kernel_size:
            # In the event the output Blackman-Harris kernel is larger than the 
            # original input kernel.
            pass
        else:
            # If the Blackman-Harris kernel is not larger change the kernel size.
            kernel_size = kernel_size_nu

    elif kernel == 'natural':
        # Natural weight cube.
        print('No natural gridding yet.')

        return None,None,None

    elif kernel == 'uniform':
        # Uniform weights cube.
        print('No uniform gridding yet.')

        return None,None,None

    # Origin centred u and v vectors.)
    
    u_vec_O = (np.arange(kernel_size) - (kernel_size-1)/2)*du
    v_vec_O = (np.arange(kernel_size) - (kernel_size-1)/2)*du

    # Calculate the u and v 2D grids.
    u_temp_arr, v_temp_arr = np.meshgrid(u_vec_O,v_vec_O)

    # Calculate the weights cube.
    weights_cube = func(u_temp_arr,v_temp_arr,size,u_shift_vec,v_shift_vec,method='square')
    #weights_cube = func(u_temp_arr,v_temp_arr,size,u_shift_vec,v_shift_vec,method='radial')

    return weights_cube


def grid(grid_arr, u_coords, v_coords, vis, u_vec, v_vec, 
        kernel_size=51, sig_grid=2.41, kernel='gaussian',test_cond=False):
    """
    Gaussian and Blackman-Harris kernel gridder. 

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
    """
    # Weight array, we will divide the entire container array by this.
    weights_arr = np.zeros(np.shape(grid_arr))

    # Initialising the grid centre index vectors.
    u_cent_ind_vec = np.zeros(u_coords.shape).astype('int')
    v_cent_ind_vec = np.zeros(v_coords.shape).astype('int')

    # Resolution required for both weighting cases.
    #delta_u = np.abs(u_grid[0,1] - u_grid[0,0])
    delta_u = np.abs(u_vec[1] - u_vec[0])
    uv_max = np.max(u_vec)

    # sig_grid should already be in wavelengths.
    sig_u = sig_grid 

    #
    ## Legacy code, 27/9/22. 
    #
    #for ind in range(len(vis)):
    #    u_cent_ind,v_cent_ind = Osiris.find_closest_xy(u_coords[ind],v_coords[ind],
    #                        u_vec,v_vec)

    #    u_cent_ind_vec[ind] = u_cent_ind
    #    v_cent_ind_vec[ind] = v_cent_ind
    
    # Nearest grid centre can be found by rounding down the u_coords.
    u_grid_cent_vec = np.rint(u_coords/delta_u)*delta_u
    v_grid_cent_vec = np.rint(v_coords/delta_u)*delta_u

    # Can use the grid coordinate to determine the index.
    u_cent_ind_vec = ((u_grid_cent_vec + uv_max)/delta_u).astype(int)
    v_cent_ind_vec = ((v_grid_cent_vec + uv_max)/delta_u).astype(int)

    # Calculating the shifted grid.
    u_shift_vec = u_coords - u_grid_cent_vec
    v_shift_vec = v_coords - v_grid_cent_vec

    weights_cube = calc_weights_cube(u_shift_vec,v_shift_vec,delta_u,sig_u,
                kernel_size=kernel_size,kernel=kernel)
    

    # Some weighting schemes (Blackman-Harris) change the number of kernel pixels.
    kernel_size = weights_cube.shape[0]

    # Determining the index ranges:
    min_u_ind_vec = u_cent_ind_vec - int(kernel_size/2)
    max_u_ind_vec = u_cent_ind_vec + int(kernel_size/2) + 1
    min_v_ind_vec = v_cent_ind_vec - int(kernel_size/2)
    max_v_ind_vec = v_cent_ind_vec + int(kernel_size/2) + 1

    # Looping through each visibility.
    for i in range(len(vis)):

        temp_weights = weights_cube[:,:,i]

        # Adding Gaussian weights to weights arr.
        weights_arr[min_v_ind_vec[i]:max_v_ind_vec[i], min_u_ind_vec[i]:max_u_ind_vec[i]] = \
            weights_arr[min_v_ind_vec[i]:max_v_ind_vec[i], min_u_ind_vec[i]:max_u_ind_vec[i]] + temp_weights

        # Adding gridded visibilities.
        grid_arr[min_v_ind_vec[i]:max_v_ind_vec[i], min_u_ind_vec[i]:max_u_ind_vec[i]] = \
            grid_arr[min_v_ind_vec[i]:max_v_ind_vec[i], min_u_ind_vec[i]:max_u_ind_vec[i]] + vis[i]*temp_weights

        # Test gridded visibilities.
        #grid_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind] = \
        #    grid_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind] + temp_gauss_weights*(1 + 0j)

    
    # Performing the weighted average.
    grid_arr[weights_arr > 0.0] = grid_arr[weights_arr > 0.0]/weights_arr[weights_arr > 0.0]


    if sig_u < delta_u:
        # Safeguard against nan, and ing value swhen sig < du.
        grid_arr.real[np.isinf(grid_arr.real)] = 0
        grid_arr.real[np.isnan(grid_arr.real)] = 0
        grid_arr.imag[np.isnan(grid_arr.imag)] = 0
        grid_arr.imag[np.isinf(grid_arr.imag)] = 0
    
    if test_cond:
        # Testing the grid kernel size.
        name = 'gridding-kernel-sig-{0}-v2'.format(np.round(sig_grid,2))
        out_path = '/home/jaiden/Documents/Skewspec/output/'
        np.savez_compressed(out_path + name, grid_arr = temp_weights, du = delta_u, \
            u = u_coords[i], v = v_coords[i])
    else:
        pass

    return grid_arr, weights_arr


def grid_cube(u_coords_arr,v_coords_arr,vis_arr,u_grid,v_grid,\
    grid_arr_cube,vis_weights_cube,weighting='gaussian',kernel_size=51,sig_grid=2.41):
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

        #grid_gaussian(grid_arr, u_coords, v_coords, vis, u_vec, v_vec)
        grid_arr_cube[:,:,i],vis_weights_cube[:,:,i] = grid(grid_arr_cube[:,:,i], \
            u_coords, v_coords, vis_vec, u_vec, v_vec, \
            kernel_size=kernel_size, sig_grid=sig_grid, kernel=weighting)

    return grid_arr_cube, vis_weights_cube
