#!/usr/bin/python

__author__ = "Jaiden Cook, Jack Line"
__credits__ = ["Jaiden Cook","Jack Line"]
__version__ = "0.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"

import numpy as np
import Osiris

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
    
        # These should be the indices of the coordinates closest to the baseline. 
        # These coordinates should line up with the kernel.
        temp_u_ind, temp_v_ind = Osiris.find_closest_xy(u[i],v[i],u_vec,v_vec)

        # Determining the index ranges:
        min_u_ind = temp_u_ind - int(kernel_size/2)
        max_u_ind = temp_u_ind + int(kernel_size/2) + 1
        min_v_ind = temp_v_ind - int(kernel_size/2)
        max_v_ind = temp_v_ind + int(kernel_size/2) + 1

        # Creating temporary u and v arrays.
        u_temp_arr = u_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind]
        v_temp_arr = v_arr[min_v_ind:max_v_ind, min_u_ind:max_u_ind]

        temp_gauss_weights = Osiris.gaussian_kernel(u_temp_arr, v_temp_arr, sig_u, sig_v, 
                                                    u[i], v[i])

        # Might have to define a visibility subset that is larger.
        # Defining the visibility subset:
        vis_sub = vis_true[min_v_ind:max_v_ind, min_u_ind:max_u_ind]
        
        # Weighted average degridded visibilitiy.
        vis_deg[i] = np.sum(vis_sub*temp_gauss_weights)/np.sum(temp_gauss_weights)
        #vis_deg[i] = np.sum(vis_sub*temp_gauss_weights)#/np.sum(temp_gauss_weights)

    #print(np.sum(temp_gauss_weights))
    #vis_deg = vis_deg/len(vis_deg)

    return vis_deg

def Vis_degrid(kernel,u_vec,v_vec,u_coords,v_coords,vis_true,w=None,phase_cond=False,
               w_ker_sample=None):
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
        phase_cond : bool
            Applies phase offset for (u,v) coord difference between pixel grid.
        skew_cond : bool
            Default False. Used when calculating skew spectrum. Squares the sky-kernel.
        w_ker_sample : int
            Default None. If not none, and is an integer, save that degrid kernel. For testing purposes. 
        

        Returns
        -------
        Weighted average of visibilities, corresponding to (u,v) points.
    """

    # Initialising the grid centre index vectors.
    u_cent_ind_vec = np.zeros(u_coords.shape).astype('int')
    v_cent_ind_vec = np.zeros(v_coords.shape).astype('int')

    # Resolution required for both weighting cases.
    #delta_u = np.abs(u_grid[0,1] - u_grid[0,0])
    delta_u = np.abs(u_vec[1] - u_vec[0])
    uv_max = np.max(u_vec)

    # Need to change some parameters here. 
    kernel_size = len(kernel.kernel)

    # Initialising the new deridded visibility array:
    vis_deg = np.zeros(len(u_coords),dtype=complex)

    # Setting the kernel to w=0, this is the default for no w-projection. If w != None then
    # the w-kernel is calculated and overwritten in the for loop below.
    kernel.calc_w_kernel(w=0.0)

    # Nearest grid centre can be found by rounding down the u_coords.
    u_grid_cent_vec = np.rint(u_coords/delta_u)*delta_u
    v_grid_cent_vec = np.rint(v_coords/delta_u)*delta_u

    # Can use the grid coordinate to determine the index.
    u_cent_ind_vec = ((u_grid_cent_vec + uv_max)/delta_u).astype(int)
    v_cent_ind_vec = ((v_grid_cent_vec + uv_max)/delta_u).astype(int)

    # Calculating the shifted grid.
    if phase_cond:
        # Condition if phase offset is true.
        u_shift_vec = u_coords - u_grid_cent_vec
        v_shift_vec = v_coords - v_grid_cent_vec
    else:
        # Default condition don't return the offsets.
        u_shift_vec = np.zeros(len(u_coords))
        v_shift_vec = np.zeros(len(u_coords))

    # Determining the index ranges:
    min_u_ind_vec = u_cent_ind_vec - int(kernel_size/2)
    max_u_ind_vec = u_cent_ind_vec + int(kernel_size/2) + 1
    min_v_ind_vec = v_cent_ind_vec - int(kernel_size/2)
    max_v_ind_vec = v_cent_ind_vec + int(kernel_size/2) + 1


    for i in range(len(u_coords)):
    
        # Might have to define a visibility subset that is larger.
        # Defining the visibility subset:
        vis_sub = vis_true[min_v_ind_vec[i]:max_v_ind_vec[i], min_u_ind_vec[i]:max_u_ind_vec[i]]
        #vis_sub = vis_true[min_u_ind:max_u_ind, min_v_ind:max_v_ind]
        
        if np.any(w):
            # If vector of w values is given.
            # Calculating the w-kernel.
            #kernel.calc_w_kernel(w[i],u_off,v_off)
            kernel.calc_w_kernel(w[i],-u_shift_vec[i],-v_shift_vec[i])

        else:
            #kernel.calc_w_kernel(0.0,u_off,v_off)
            kernel.calc_w_kernel(0.0,-u_shift_vec[i],-v_shift_vec[i])

        # Weighted average degridded visibilitiy.
        temp_vis = np.average(vis_sub,weights=kernel.w_kernel)
        #vis_deg[i] = temp_vis.real
        vis_deg[i] = temp_vis

        # Don't need to save the sample.
        w_ker_sample = None
        if w_ker_sample != None:
            # Saving a sample of the deridding kernel for testing purposes. 

            w_ker_sample = int(w_ker_sample)

            if w_ker_sample == i:

                print('Saving degridding kernel w=%5.2f' % (w[i]))
                # Save visibilities. Useful for testing purposes. 
                out_path = '/home/jaiden/Documents/EoR/SNR-Pipeline/output/'
                
                import time
                # Need a way to differentiate between visibilities for multiple runs.
                # In the future this will be ammended to a more common sense name.
                time_current = round(time.time())

                name = 'degrid-ker-w{1}-{0}'.format(time_current,np.round(w[i],2))
                np.savez_compressed(out_path + name, w_kernel = kernel.w_kernel)

                print('Degridding kernel saved at %s.npz' % (out_path + name))
            else:
                pass
        else:
            pass

    return vis_deg


class w_kernel():
    """
    Creates the input arrays for calculating the w-sky-kernel, and saves them. 
    Takes input w-terms and outputs the corresponding w-sky-kernel.

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
        """
        Returns the w-sky-kernel.

        Parameters
        ----------
        w : float
            Baseline w-coordinate.
        u_off : float
            Offset in u-coords/
        v_off : float
            Offset in v-coords
        
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
            offset_grid = np.exp(2j*np.pi*(u_off*self.l_grid + v_off*self.m_grid))
            #offset_grid = np.exp(-2j*np.pi*(v_off*self.l_grid + u_off*self.m_grid))
            # Calculating.
            w_sky_ker = offset_grid*np.exp(2j*np.pi*w*(self.n_grid - 1))*self.kernel
            #w_sky_ker = np.exp(-2j*np.pi*w*(self.n_grid - 1))*self.kernel
        else:
            # Default don't phase rotate relative to the offsets.
            w_sky_ker = np.exp(-2j*np.pi*w*(self.n_grid - 1))*self.kernel

        self.w_sky_ker = w_sky_ker

        # Default Power spectrum degridding case
        u_grid, v_grid, w_kernel = Osiris.Visibilities_2D(w_sky_ker,L,M,N,norm='forward')

        # Setting attributes:
        self.u_grid = u_grid
        self.v_grid = v_grid

        # Setting and normalising the w-kernel.
        self.w_kernel = w_kernel # 17/7/22 -- depreciated
        ##self.w_kernel = w_kernel/np.abs(np.sum(w_kernel)) # old

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
                Osiris.Plot_img(self.w_sky_ker.real,self.l_grid,self.m_grid,cmap='viridis',
                                figsize=(5,5),clab='Response',xlab=r'$l$',ylab=r'$m$',
                                title=title,**kwargs)
            elif imag_cond:
                # Plot imag part of sky kernel:
                Osiris.Plot_img(self.w_sky_ker.imag,self.l_grid,self.m_grid,cmap='viridis',
                                figsize=(5,5),clab='Response',xlab=r'$l$',ylab=r'$m$',
                                title=title,**kwargs)
        elif ker == 'vis':
            if real_cond:
                # Plot real part of the w-kernel:
                Osiris.Plot_img(self.w_kernel.real,self.u_grid,self.v_grid,cmap='viridis',
                                figsize=(5,5),clab='Response',xlab=r'$u\,[\lambda]$',
                                ylab=r'$v\,[\lambda]$',title=title,**kwargs)
            elif imag_cond:
                # Plot imag part of the w-kernel:
                Osiris.Plot_img(self.w_kernel.imag,self.u_grid,self.v_grid,cmap='viridis',
                                figsize=(5,5),clab='Response',xlab=r'$u\,[\lambda]$',
                                ylab=r'$v\,[\lambda]$',title=title,**kwargs)

