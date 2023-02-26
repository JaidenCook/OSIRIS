#!/usr/bin/python

__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"

# Generic stuff:
import os,sys
import time
import warnings

from traitlets.traitlets import default
warnings.filterwarnings("ignore")

# Array stuff:
import numpy as np
#warnings.simplefilter('ignore', np.RankWarning)

# Plotting stuff:
import matplotlib.pyplot as plt
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

# MWA beam stuff
from mwa_pb import primary_beam as pb

sys.path.append(os.path.abspath("/home/jaiden/Documents/EoR/OSIRIS"))
import Osiris
import Osiris_spec_nu

def KDEpy_1D_scaled(data,weights=None,x=None,bw_method='ISJ',N_dim=256,verbose=False):
    """
    For values with an order of magnitude of 6 or greater KDEpy FFTKDE fails due 
    to lack of finite support. The solution is to scale the data so the bandwidth is 1. 
    The solution is found in https://github.com/tommyod/KDEpy/issues/81. This is the
    same solution implemented in KDEpy_2D_scaled.
    
    Parameters
    ----------
    data : array : float
        1D array of length (N_samples) containing the X.
    weights : array : float
        1D array of shape (N_samples,) containing the X samples weights.
        Default is None.
    x : array : float
        2D array with diminsions (N,N) calculating the evenly spaced x grid.
    bw_method : string
        Bandwidth method, default is ISJ, other options are 'silvermans' and 'scotts' method.
    N_dim : int
        Number of evaluations. The default is 256. This is given when x=None. This specifies the
        grid size. 
    verbose : bool
        If True print outputs, useful for testing. 

    Returns
    -------
    y : array : float
        1D array of the p(x) KDE estimated values.
    """
    from KDEpy.bw_selection import silvermans_rule, improved_sheather_jones,scotts_rule
    from KDEpy.FFTKDE import FFTKDE

    # Compute the BW, scale, then scale back
    if bw_method == 'silverman':
        data_temp = data[data != 0.0]
        #bw = silvermans_rule(data[:,None])
        bw = silvermans_rule(data_temp[:,None])
        #data[:, [0]]

    elif bw_method == 'scott':
        data_temp = data[data != 0.0]
        #bw = scotts_rule(data[:,None])
        bw = scotts_rule(data_temp[:,None])

    else:
        # ISJ is the default method. 
        data_temp = data[data != 0.0]
        #bw = improved_sheather_jones(data[:,None])
        bw = improved_sheather_jones(data_temp[:,None])

    if bw == 0.0:

        raise ValueError('Bandwidth is zero...')

    # Data is rescaled by the bandwidth. 
    data_scaled = data / np.array([bw])

    if verbose:
        print('Bw method = %s' % bw_method)
        print('bw = %5.3e' % bw)

    if np.any(x):
        # Case when grid is provided.
        #
        # Scaling the positions.
        x_scaled = x / np.array([bw])

        y_scaled = FFTKDE(bw=1).fit(data_scaled,weights=weights).evaluate(x_scaled)
        #y_scaled = FFTKDE(bw=1).fit(data_scaled*weights).evaluate(x_scaled)
        y = y_scaled / (bw)

        x = x_scaled * bw

    else:
        # If a grid is not provided use auto_grid feature to calculate the 2D KDE.
        x_scaled, y_scaled = FFTKDE(bw=1).fit(data_scaled,weights=weights).evaluate(N_dim)
        
        x = x_scaled * np.array([bw])
        y = y_scaled / (bw)

    return y

def KDEpy_2D_scaled(data,weights=None,xx=None,yy=None,bw_method='ISJ',N_dim=128,verbose=False):
    """
    This method is taken from https://github.com/tommyod/KDEpy/issues/81. This allows for determining
    the bandwidth using ISJ, Silvermann, or Scott. KDEpy currently doesn't support this for 2D kernels
    or larger.
    
    Parameters
    ----------
    data : array : float
        2D array of shape (N_samples,2) containing the X and Y random samples.
    weights : array : float
        2D array of shape (N_samples,2) containing the X and Y random samples weights.
        Default is None.
    xx : array : float
        2D array with diminsions (N,N) calculating the evenly spaced x grid.
    yy : array : float
        2D array with diminsions (N,N) calculating the evenly spaced y grid.
    bw_method : string
        Bandwidth method, default is ISJ, other options are 'silvermans' and 'scotts' method.
    verbose : bool
        If True print ouputs. This is useful for testing purposes.

    Returns
    -------
    zz : array : float
        2D array of the p(x,y) KDE estimated values.
    """
    from KDEpy.bw_selection import silvermans_rule, improved_sheather_jones,scotts_rule
    from KDEpy.FFTKDE import FFTKDE

    # Compute the BW, scale, then scale back
    if bw_method == 'silverman':
        bw1 = silvermans_rule(data[:, [0]][data[:, [0]] != 0.0][:,None])
        bw2 = silvermans_rule(data[:, [1]][data[:, [1]] != 0.0][:,None])
        
    elif bw_method == 'scott':
        bw1 = scotts_rule(data[:, [0]][data[:, [0]] != 0.0][:,None])
        bw2 = scotts_rule(data[:, [1]][data[:, [1]] != 0.0][:,None])

    else:
        # ISJ is the default method. 
        bw1 = improved_sheather_jones(data[:, [0]][data[:, [0]] != 0.0][:,None])
        bw2 = improved_sheather_jones(data[:, [1]][data[:, [1]] != 0.0][:,None])

    # Axes are swapped, ordering with KDEpy is different. 
    # If not swapped, ValueError is occassionally raised related to grid points
    # not within the data range. 
    data = np.roll(data,1,axis=1)
    
    if verbose:
        print('Bw method = %s' % bw_method)
        print('Bandwidths:')
        print('bw1 = %5.3e' % bw1)
        print('bw2 = %5.3e' % bw2)


    if np.any(weights):
        #weights_nu = np.empty(weights.shape)
        #weights_nu[:,0] = weights[:,1]
        #weights_nu[:,1] = weights[:,0]

        # 2D Weights are the average of the two 1D weights. 
        #weights_nu = 0.5*(weights_nu[:,0] + weights_nu[:,1])
        weights_nu = 0.5*(weights[:,0] + weights[:,1])
        
    else:
        weights_nu = None

    # Data is rescaled by the bandwidth. 
    data_scaled = data / np.array([bw1, bw2])

    
    if np.any(xx) and np.any(yy):
        # Case when grid is provided.

        # Flattening the arrays for formatting. 
        x = xx.flatten()
        y = yy.flatten()

        # Creating a positions array.
        # This has to be sorted in a specific way to work through KDEpy.FFTKDE method. 
        positions = np.empty((len(x),2))

        positions[:,0] = y # Dims are swapped.
        positions[:,1] = x

        # Scaling the positions.
        positions_scaled = positions / np.array([bw1, bw2])

        z_scaled = FFTKDE(bw=1).fit(data_scaled,weights=weights_nu).evaluate(positions_scaled)
        
        z = z_scaled / (bw1 * bw2)
        zz = z.reshape(xx.shape)

    else:
        # If a grid is not provided use auto_grid feature to calculate the 2D KDE.
        x_scaled, z_scaled = FFTKDE(bw=1).fit(data_scaled,weights=weights_nu).evaluate((N_dim, N_dim))
        x = x_scaled * np.array([bw1, bw2])

        z = z_scaled / (bw1 * bw2)
        zz = z.reshape((N_dim,N_dim))

    return zz

def Scatter_hist2D(X_samp,Y_samp,weights_arr=None,figaxs=None,figsize=(11, 10)
    ,pxlab=None,pylab=None,method='scatter',**kwargs):
    """
    Plot 2D scatter plot, with X and Y histograms.

    Parameters
    ----------
        X_samp : array
            ND array of mean values. The length of this array should be equal to the
            number of Gaussian components.
        Y_samp : array 
            ND array of sigma values. The length of this array should be equal to the
            number of Gaussian components.
        figaxs : tuple 
            Tuple containing fig and axs objects. Default is None. 
        figsize : tuple
            Tuple containing figure size. Default is (11,10).
        method : string
            2D plotting method, default is 'scatter', other choices are 'hist2d' and 'hexbin'.
        **kwargs : 
            plt.hist kwarg arguments. 
        

    Returns
    -------
    None
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable


    if figaxs:
        fig = figaxs[0]
        axs = figaxs[1]
    else:
        fig, axs = plt.subplots(1, figsize = figsize, dpi=75)

    if np.any(weights_arr):
        # If any weights.
        X_weights = weights_arr[:,0]
        Y_weights = weights_arr[:,1]
        #weights2D = X_weights*Y_weights

        # 2D weights are the average of the two sets of 1D weights. 
        weights2D = 0.5*(X_weights + Y_weights)

    else:
        # Default is no weights.
        X_weights = None
        Y_weights = None
        weights2D = None

    # Calculate the number of bins.
    bins1D = int(np.sqrt(len(X_samp)))

    # If the number of bins is less than 100, set the default to 100.
    if bins1D < 100:
        bins1D = 100
    else:
        pass

    # Selecting 2D plotting method:
    if method == 'hist2d':
        #
        bins2D = int(np.sqrt(0.5*len(X_samp)))
        if bins2D > 100:
            bins2D = 100
        #cmap = plt.cm.BuPu
        cmap = 'Blues'
        #axs.hist2d(X_samp,Y_samp,bins=(bins2D, bins2D), cmap=cmap,density=True)
        axs.hist2d(X_samp,Y_samp,bins=(bins2D,bins2D),cmap=cmap,
                    density=True,weights=weights2D)
    elif method == 'hexbin':
        # Hexbin method. No weights method for hexbin.
        bins2D = int(np.sqrt(0.5*len(X_samp)))
        #cmap = plt.cm.BuPu
        cmap = 'Blues'
        axs.hexbin(X_samp,Y_samp,gridsize=bins2D, cmap=cmap)

        # Default is no weights.
        X_weights = None
        Y_weights = None
    else:
        # Scatter plot is the default method. 
        axs.scatter(X_samp,Y_samp)

        # Default is no weights.
        #X_weights = None
        #Y_weights = None
    
    axs.set_xlabel(r'$X$',fontsize=18)
    axs.set_ylabel(r'$Y$',fontsize=18)

    # Set aspect of the main axes.
    #ax.set_aspect(1.)

    # create new axes on the right and on the top of the current axes
    divider = make_axes_locatable(axs)

    # below height and pad are in inches
    ax_histx = divider.append_axes("top", 2, pad=0.2, sharex=axs)
    ax_histy = divider.append_axes("right", 2, pad=0.2, sharey=axs)

    # make some labels invisible
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    if kwargs:
        ax_histx.hist(X_samp,bins=bins1D,density=True,weights=X_weights,
            label=pxlab,**kwargs)
        ax_histy.hist(Y_samp,bins=bins1D,density=True,weights=Y_weights,orientation='horizontal',
            label=pylab,**kwargs)
    else:
        ax_histx.hist(X_samp,bins=bins1D,density=True,weights=X_weights,
            histtype='stepfilled',alpha=0.5,align='mid',edgecolor='k',label=pxlab,)
        ax_histy.hist(Y_samp,bins=bins1D,density=True,weights=Y_weights,orientation='horizontal',
            histtype='stepfilled',alpha=0.5,align='mid',edgecolor='k',label=pylab,)

    ax_histx.legend(fontsize=18)
    ax_histy.legend(fontsize=18)    

    ax_histx.set_ylabel(r'$p_X(x)$',fontsize=18)
    ax_histy.set_xlabel(r'$p_Y(y)$',fontsize=18)

    plt.show()

def Plot_joint_marginal_dists(xx,yy,x,y,pxy,px,py,figaxs=None,figsize=(11, 10)
    ,pxlab=None,pylab=None,logcond=False,**kwargs):
    """
    Plot 2D joint KDE distribution, and the 1D marginal distributions.

    Parameters
    ----------
        xx : array
            2D array of x grid values. For 2D KDE contour lines.
        yy : array
            2D array of y grid values. For 2D KDE contour lines.
        x : array
            1D array of x grid values. For 1D KDE.
        y : array
            1D array of y grid values. For 1D KDE.
        pxy : array 
            2D KDE values.
        px : array
            1D x KDE values.
        py : 
            1D y KDE values.
        figaxs : tuple 
            Tuple containing fig and axs objects. Default is None. 
        figsize : tuple
            Tuple containing figure size. Default is (11,10).
        **kwargs : 
            plt.hist kwarg arguments. 
        

    Returns
    -------
    None
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable


    if figaxs:
        fig = figaxs[0]
        axs = figaxs[1]
    else:
        fig, axs = plt.subplots(1, figsize = figsize, dpi=75)

    if logcond:
        # If log scale is true. This can have some weird 2D plotting results.
        cfset = axs.contourf(xx, yy, np.log10(pxy), cmap='Blues')
        cset = axs.contour(xx, yy, np.log10(pxy), colors='k')
        axs.clabel(cset, inline=1, fontsize=10)
    else:
        cfset = axs.contourf(xx, yy, pxy, cmap='Blues')
        cset = axs.contour(xx, yy, pxy, colors='k')
        axs.clabel(cset, inline=1, fontsize=10)

    # Selecting 2D plotting method:
    
    axs.set_xlabel(r'$X$',fontsize=18)
    axs.set_ylabel(r'$Y$',fontsize=18)

    # Set aspect of the main axes.
    #ax.set_aspect(1.)

    # create new axes on the right and on the top of the current axes
    divider = make_axes_locatable(axs)

    # below height and pad are in inches
    ax_histx = divider.append_axes("top", 2, pad=0.2, sharex=axs)
    ax_histy = divider.append_axes("right", 2, pad=0.2, sharey=axs)

    # make some labels invisible
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    if kwargs:
        ax_histx.plot(x, px,label=pxlab,**kwargs)
        ax_histy.plot(py, y,label=pylab,**kwargs)
    else:
        ax_histx.plot(x, px,label=pxlab, lw=2)
        ax_histy.plot(py, y,label=pylab, lw=2)

    ax_histx.set_ylabel(r'$p_X(x)$',fontsize=18)
    ax_histy.set_xlabel(r'$p_Y(y)$',fontsize=18)

    ax_histx.legend(fontsize=18)
    ax_histy.legend(fontsize=18)

    if logcond:
        # Some distributions may be preferable to be plotted in log scale.
        ax_histx.set_yscale('log')
        ax_histy.set_xscale('log')
    else:
        pass

    plt.show()

def Spherical(k_r_arr,real_vis_cube_0,weights_cube_0,real_vis_cube_1,weights_cube_1,
            N_bins=50,log_bin_cond=False,kr_min=None,kr_max=None,bw='scott'):
    """
    Calculates the 1D spherical mutual information as a function of k.
            
        Parameters
        ----------
        self : object
            Power object contains u and v arrays, as well as the observation redshift.
        kb : float
            Boltzman's constant.
        nu_21 : float
            21cm frequency in Hz.
        c : float
            Speed of light km/s.
        
        Returns
        -------
    """
    import sys

    
    if kr_min:
        # User can manually input a kr min.
        kr_min = float(kr_min)
        #kr_min = 0
    else:
        kr_min = np.nanmin(k_r_arr[k_r_arr > 0.0])
        #kr_min = 0

    if kr_max:
        # User can manually input a kr max.
        kr_max = float(kr_max)
    else:
        kr_max = np.nanmax(k_r_arr)

    print('k_r_min = %5.3f' % kr_min)
    print('k_r_max = %5.3f' % kr_max)

    if log_bin_cond:
        # Logarithmically spaced bins, creates uniform bin widths in log space plots.
        # Binning conditions are different for log bins.
        if N_bins == 60:
            # Default number for lin bins is 60. If still 60 then set new default.
            N_bins = 15
        else:
            # If not default N_bins = 60, user has input new number.
            pass

        # Log-linear bins.
        log_kr_min = np.log10(kr_min)
        log_kr_max = np.log10(kr_max)
        
        # Increments.
        dlog_k = (log_kr_max - log_kr_min)/(N_bins + 1)

        k_r_bins = np.logspace(log_kr_min - dlog_k/2,log_kr_max + dlog_k/2,N_bins + 1)
        print('dlog_k = %5.3e' % dlog_k)

    else:
        
        # Increments.
        dk = (kr_max - kr_min)/(N_bins + 1)
        k_r_bins = np.linspace(kr_min,kr_max,N_bins + 1)

        print('dk = %5.3f' % dk)


    #np.savez_compressed('/home/jaiden/Documents/Skewspec/output/' + 'k_r_bins', k_r_bins = k_r_bins)
    #print('Bin edges saved for testing purposes...')

    print('N_bins = %s' % N_bins)

    start0 = time.perf_counter()
    mutual_inf_k = np.zeros(N_bins)
    kr_vec = np.zeros(N_bins)


    for i in range(len(k_r_bins)-1):

        # Progress bar:
        if (i+1) % 10 == 0:
            sys.stdout.write("\rBins processed: {0}/{1}".format((i+1),N_bins))
            sys.stdout.flush()
        elif (i+1) == N_bins:
            # Last iteration.
            sys.stdout.write("\rBins processed: {0}/{0}\n".format(N_bins))
            sys.stdout.flush()
        else:
            pass

        # Calculating the radius:
        if log_bin_cond:
            kr_vec[i] = 10**(0.5*(np.log10(k_r_bins[i+1]) + np.log10(k_r_bins[i])))
        else:
            kr_vec[i] = ((k_r_bins[i+1] + k_r_bins[i])/2.0)

        # Defining the shell array index:
        shell_ind = np.logical_and(k_r_arr >= k_r_bins[i], k_r_arr <= k_r_bins[i+1])

        # Getting the shell and weight values.
        EoR0_shell_vals = real_vis_cube_0[shell_ind]
        EoR0_shell_weights = weights_cube_0[shell_ind]

        # Getting the shell and weight values.
        EoR1_shell_vals = real_vis_cube_1[shell_ind]
        EoR1_shell_weights = weights_cube_1[shell_ind]

        ###
        # Some cells only have values in one grid and not the other. 
        # We will subset these out.
        ###

        Shell_ind_non_zero = (EoR0_shell_vals != 0.0)*(EoR1_shell_vals != 0.0)

        # Getting the shell and weight values.
        EoR0_shell_vals = EoR0_shell_vals[Shell_ind_non_zero]
        EoR0_shell_weights = EoR0_shell_weights[Shell_ind_non_zero]

        # Getting the shell and weight values.
        EoR1_shell_vals = EoR1_shell_vals[Shell_ind_non_zero]
        EoR1_shell_weights = EoR1_shell_weights[Shell_ind_non_zero]

        ##
        #
        ##
        std_fac = 0.01

        x_min = np.min(EoR0_shell_vals)-std_fac*np.std(EoR0_shell_vals)
        x_max = np.max(EoR0_shell_vals)+std_fac*np.std(EoR0_shell_vals)
        y_max = np.max(EoR1_shell_vals)+std_fac*np.std(EoR1_shell_vals)
        y_min = np.min(EoR1_shell_vals)-std_fac*np.std(EoR1_shell_vals)

        nside = 1000

        x = np.linspace(x_min,x_max,nside)
        y = np.linspace(y_min,y_max,nside)

        vals_x = KDEpy_1D_scaled(EoR0_shell_vals,weights=EoR0_shell_weights,x=x,bw_method=bw)
        vals_y = KDEpy_1D_scaled(EoR1_shell_vals,weights=EoR1_shell_weights,x=y,bw_method=bw)

        xx,yy = np.meshgrid(x,y)

        Vals_vec = np.empty((len(EoR0_shell_vals),2))

        Vals_vec[:,0] = EoR0_shell_vals
        Vals_vec[:,1] = EoR1_shell_vals

        dx = np.abs(x[1]-x[0])
        dy = np.abs(y[1]-y[0])

        dx2D = np.abs(x[1]-x[0])
        dy2D = np.abs(y[1]-y[0])


        # Initialising a weights array.
        weights = np.empty((len(EoR1_shell_weights),2))

        # Assigning the weight values to the array.
        weights[:,0] = EoR0_shell_weights
        weights[:,1] = EoR1_shell_weights

        zz = KDEpy_2D_scaled(Vals_vec,weights,xx,yy,bw_method=bw)

        if i == 50:
            scale = 0.8
            fig,axs = plt.subplots(1,figsize=(scale*12,scale*10))

            figaxs = (fig,axs)

            Plot_joint_marginal_dists(xx,yy,x,y,zz,vals_x,vals_y,pxlab='EoR0',pylab='EoR1',
                figaxs=figaxs,lw=2.5,logcond=False)

        diff_ent_X = MI_metric.diff_ent_1D(vals_x,dx)
        diff_ent_Y = MI_metric.diff_ent_1D(vals_y,dy)
        diff_ent_pxy = MI_metric.diff_ent_2D(zz,dx2D,dy2D)

        mutual_inf_k[i] = MI_metric.mutual_information(diff_ent_X,diff_ent_Y,diff_ent_pxy)


    end0 = time.perf_counter()
    
    print('1D MI calctime = %5.3f s' % (end0-start0))

    return mutual_inf_k,kr_vec


class MI_metric:
    """
    Mutual information metrics class. 

    Methods
    -------
    diff_ent_1D(px,dx)
        Calculate the 1D differential entropy.
    diff_ent_2D(pxy,dx,dy)
        Calculate the 2D differential entropy.
    mutual_information(diff_ent_x,diff_ent_y,diff_ent_xy)
        Calculate the mutual information.
    variation_information(diff_ent_x,diff_ent_y,diff_ent_xy)
        Calculate the variation of information.
    normalised_mutual_information(diff_ent_x,diff_ent_y,diff_ent_xy)
        Calculate the normalised mutual information.
    information_quality_ratio(diff_ent_x,diff_ent_y,diff_ent_xy)
        Calculate the information quality ratio.
    """

    @staticmethod
    def diff_ent_1D(px,dx):
        """
        Calculate the 1D differential entropy. Uses a simple Reimann sum to 
        estimate the differential entropy. There are scipy functions that do the
        1D calculations.

            Parameters
            ----------
            px : numpy array
                1D vector of probability density values.
            dx : float
                Grid size of the 1D vector. 
            diff_ent_xy : float
                Differential entropy for marginal XY distribution.
            
            Returns
            -------
        """

        return np.sum(-px*np.log(px))*dx

    @staticmethod
    def diff_ent_2D(pxy,dx,dy):
        """
        Calculate the 1D differential entropy. Uses a simple Reimann sum to 
        estimate the differential entropy. There are scipy functions that do the
        1D calculations.

            Parameters
            ----------
            px : numpy array
                1D vector of probability density values.
            dx : float
                Grid size of the X vector. 
            dy : float
                Grid size of the Y vector. 
            
            
            Returns
            -------
        """

        return np.sum(-pxy*np.log(pxy))*dx*dy


    @staticmethod
    def mutual_information(diff_ent_x,diff_ent_y,diff_ent_xy):
        """
        Mutual information. 
            
            Parameters
            ----------
            diff_ent_x : float
                Differential entropy for random variable X.
            diff_ent_y : float
                Differential entropy for random variable Y.
            diff_ent_xy : float
                Differential entropy for marginal XY distribution.
            
            Returns
            -------
        """
        return diff_ent_x + diff_ent_y - diff_ent_xy

    @staticmethod
    def variation_information(diff_ent_x,diff_ent_y,diff_ent_xy):
        """
        Distance measure, satisfies the triangle inequality.

            Parameters
            ----------
            diff_ent_x : float
                Differential entropy for random variable X.
            diff_ent_y : float
                Differential entropy for random variable Y.
            diff_ent_xy : float
                Differential entropy for marginal XY distribution.
            
            Returns
            -------
        """
        return 2*diff_ent_xy - diff_ent_x - diff_ent_y

    @staticmethod
    def normalised_mutual_information(diff_ent_x,diff_ent_y,diff_ent_xy):
        """
        Analogous to Pearson's correlation coefficient. Returns the normalised mutual 
        information.

            Parameters
            ----------
            diff_ent_x : float
                Differential entropy for random variable X.
            diff_ent_y : float
                Differential entropy for random variable Y.
            diff_ent_xy : float
                Differential entropy for marginal XY distribution.
            
            Returns
            -------
        """
        MI = MI_metric.mutual_information(diff_ent_x,diff_ent_y,diff_ent_xy)
        return MI/np.sqrt(diff_ent_x*diff_ent_y)

    @staticmethod
    def information_quality_ratio(diff_ent_x,diff_ent_y,diff_ent_xy):
        """
        Quantifies the amount of information of a variable based on another variable against
        the total uncertainty H(X,Y).

            Parameters
            ----------
            diff_ent_x : float
                Differential entropy for random variable X.
            diff_ent_y : float
                Differential entropy for random variable Y.
            diff_ent_xy : float
                Differential entropy for marginal XY distribution.
            
            Returns
            -------
        """
        MI = MI_metric.mutual_information(diff_ent_x,diff_ent_y,diff_ent_xy)
        return MI/diff_ent_xy

