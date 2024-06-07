#!/usr/bin/python

__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"

# Generic stuff:
import warnings

from traitlets.traitlets import default
warnings.filterwarnings("ignore")

# Array stuff:
import numpy as np
#warnings.simplefilter('ignore', np.RankWarning)

# Plotting stuff:
import matplotlib.pyplot as plt

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


def KDEpy_1D_scaled(data,weights=None,x=None,bw_method='ISJ',Ndim=256,
                    verbose=False):
    """
    For values with an order of magnitude of 6 or greater KDEpy FFTKDE fails due 
    to lack of finite support. The solution is to scale the data so the 
    bandwidth is 1. The solution is found in 
    https://github.com/tommyod/KDEpy/issues/81. This is the same solution 
    implemented in KDEpy_2D_scaled.
    
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
        Bandwidth method, default is ISJ, other options are 'silvermans' and 
        'scotts' method.
    N_dim : int
        Number of evaluations. The default is 256. This is given when x=None. 
        This specifies the grid size. 
    verbose : bool
        If True print outputs, useful for testing. 

    Returns
    -------
    y : array : float
        1D array of the p(x) KDE estimated values.
    """
    from KDEpy.bw_selection import silvermans_rule, improved_sheather_jones
    from KDEpy.bw_selection import scotts_rule
    from KDEpy.FFTKDE import FFTKDE

    # Compute the BW, scale, then scale back
    if bw_method == 'silverman':
        dataTemp = data[data != 0.0]
        #bw = silvermans_rule(data[:,None])
        bw = silvermans_rule(dataTemp[:,None])
        #data[:, [0]]

    elif bw_method == 'scott':
        dataTemp = data[data != 0.0]
        #bw = scotts_rule(data[:,None])
        bw = scotts_rule(dataTemp[:,None])

    else:
        # ISJ is the default method. 
        dataTemp = data[data != 0.0]
        #bw = improved_sheather_jones(data[:,None])
        bw = improved_sheather_jones(dataTemp[:,None])

    if bw == 0.0:

        raise ValueError('Bandwidth is zero...')

    # Data is rescaled by the bandwidth. 
    datascld = data/np.array([bw])

    if verbose:
        print('Bw method = %s' % bw_method)
        print('bw = %5.3e' % bw)

    if np.any(x):
        # Case when grid is provided.
        #
        # Scaling the positions.
        xscaled = x/np.array([bw])

        yscaled = FFTKDE(bw=1).fit(datascld,weights=weights).evaluate(xscaled)
        #y_scaled = FFTKDE(bw=1).fit(datascld*weights).evaluate(xscaled)
        y = yscaled/bw

        x = xscaled*bw

    else:
        # If a grid is not provided use auto_grid feature to calculate the 2D KDE.
        xscaled,yscaled = FFTKDE(bw=1).fit(datascld,weights=weights).evaluate(Ndim)
        
        x = xscaled*np.array([bw])
        y = yscaled/bw

    return y

def KDEpy_2D_scaled(data,weights=None,xx=None,yy=None,bw_method='ISJ',
                    N_dim=128,verbose=False):
    """
    This method is taken from https://github.com/tommyod/KDEpy/issues/81. This 
    allows for determining the bandwidth using ISJ, Silvermann, or Scott. 
    KDEpy currently doesn't support this for 2D kernels or larger.
    
    Parameters
    ----------
    data : array : float
        2D array of shape (N_samples,2) containing the X and Y random samples.
    weights : array : float
        2D array of shape (N_samples,2) containing the X and Y random samples 
        weights. Default is None.
    xx : array : float
        2D array with diminsions (N,N) calculating the evenly spaced x grid.
    yy : array : float
        2D array with diminsions (N,N) calculating the evenly spaced y grid.
    bw_method : string
        Bandwidth method, default is ISJ, other options are 'silvermans' 
        and 'scotts' method.
    verbose : bool
        If True print ouputs. This is useful for testing purposes.

    Returns
    -------
    zz : array : float
        2D array of the p(x,y) KDE estimated values.
    """
    from KDEpy.bw_selection import improved_sheather_jones,scotts_rule
    from KDEpy.bw_selection import silvermans_rule
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

def Scatter_hist2D(X_samp,Y_samp,weights_arr=None,figaxs=None,figsize=(11,10),
                   pxlab=None,pylab=None,method='scatter',**kwargs):
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
        2D plotting method, default is 'scatter', other choices are 'hist2d' and 
        'hexbin'.
    **kwargs : 
        plt.hist kwarg arguments. 
        

    Returns
    -------
    None
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable


    if figaxs:
        _ = figaxs[0]
        axs = figaxs[1]
    else:
        _,axs = plt.subplots(1,figsize=figsize,dpi=75)

    if np.any(weights_arr):
        # If any weights.
        X_weights = weights_arr[:,0]
        Y_weights = weights_arr[:,1]
        #weights2D = X_weights*Y_weights

        # 2D weights are the average of the two sets of 1D weights. 
        weights2D = 0.5*(X_weights+Y_weights)

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
        axs.hexbin(X_samp,Y_samp,gridsize=bins2D,cmap=cmap)

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
    ax_histx = divider.append_axes("top",2,pad=0.2,sharex=axs)
    ax_histy = divider.append_axes("right",2,pad=0.2,sharey=axs)

    # make some labels invisible
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    if kwargs:
        ax_histx.hist(X_samp,bins=bins1D,density=True,weights=X_weights,
                      label=pxlab,**kwargs)
        ax_histy.hist(Y_samp,bins=bins1D,density=True,weights=Y_weights,
                      orientation='horizontal',label=pylab,**kwargs)
    else:
        ax_histx.hist(X_samp,bins=bins1D,density=True,weights=X_weights,
                      histtype='stepfilled',alpha=0.5,align='mid',edgecolor='k',
                      label=pxlab,)
        ax_histy.hist(Y_samp,bins=bins1D,density=True,weights=Y_weights,
                      orientation='horizontal',histtype='stepfilled',
                      alpha=0.5,align='mid',edgecolor='k',label=pylab,)

    ax_histx.legend(fontsize=18)
    ax_histy.legend(fontsize=18)    

    ax_histx.set_ylabel(r'$p_X(x)$',fontsize=18)
    ax_histy.set_xlabel(r'$p_Y(y)$',fontsize=18)

    plt.show()

def Plot_joint_marginal_dists(xx,yy,x,y,pxy,px,py,figaxs=None,figsize=(11,10),
                              pxlab=None,pylab=None,logcond=False,filename=None,
                              **kwargs):
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
    filename : str
        If given output png file to filename. Include path in the name.
    **kwargs : 
        plt.hist kwarg arguments. 
    

    Returns
    -------
    None
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if figaxs:
        _ = figaxs[0]
        axs = figaxs[1]
    else:
        _,axs = plt.subplots(1,figsize=figsize,dpi=75)

    if logcond:
        # If log scale is true. This can have some weird 2D plotting results.
        cfset = axs.contourf(xx,yy,np.log10(pxy),cmap='Blues')
        cset = axs.contour(xx,yy,np.log10(pxy),colors='k')
        axs.clabel(cset,inline=1,fontsize=10)
    else:
        cfset = axs.contourf(xx,yy,pxy,cmap='Blues')
        cset = axs.contour(xx,yy,pxy,colors='k')
        axs.clabel(cset,inline=1,fontsize=10)

    # Selecting 2D plotting method:
    axs.set_xlabel(r'$X$',fontsize=18)
    axs.set_ylabel(r'$Y$',fontsize=18)

    # Set aspect of the main axes.
    #ax.set_aspect(1.)

    # create new axes on the right and on the top of the current axes
    divider = make_axes_locatable(axs)

    # below height and pad are in inches
    ax_histx = divider.append_axes("top",2,pad=0.2,sharex=axs)
    ax_histy = divider.append_axes("right",2,pad=0.2,sharey=axs)

    # make some labels invisible
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    if kwargs:
        ax_histx.plot(x,px,label=pxlab,**kwargs)
        ax_histy.plot(py,y,label=pylab,**kwargs)
    else:
        ax_histx.plot(x,px,label=pxlab,lw=2)
        ax_histy.plot(py,y,label=pylab,lw=2)

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

    if filename:
        plt.savefig(filename,bbox_inches='tight')
    else:
        plt.show()

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
    calc_spherical_MI(dataX_shell,dataY_shell,dataX_weights=None,
                      dataY_weights=None,plot_cond=False,bw='scott',
                      std_fac=0.01,nside=1000)
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
        Analogous to Pearson's correlation coefficient. Returns the normalised 
        mutual information.

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
        Quantifies the amount of information of a variable based on another 
        variable against the total uncertainty H(X,Y).

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
    
    def calc_KDE_MI(dataX_shell,dataY_shell,dataX_weights=None,
                    dataY_weights=None,plot_cond=False,bw='scott',std_fac=0.01,
                    nside=1000):
        """
        Calculate the spherical mutual information from two input 3D data arrays.

        Parameters
        ----------
        dataX_shell : numpy array
            Numpy array of X spherical shell values.
        dataY_shell : numpy array
            Numpy array of Y spherical shell values.
        dataX_weights : numpy array
            Numpy array of X spherical shell value weights.
        dataY_weights : numpy array
            Numpy array of Y spherical shell value weights.
        plot_cond : bool, default=False
            If true plot the marginal distribution.
        bw : str, default='scott'
            KDE bandwidth estimation method, options are 'silverman', 'ISJ', and 
            'scott'.
        std_fac : float, default=0.01
            Padding for the X and Y grid values.
        nside : int, default=1000
            Grid size, reduce this to decrease computation at the expense of 
            accuracy.
        

        Returns
        -------
        MI_temp : float
            Mutual information of the two spherical shells dataX_shell and 
            dataY_shell.
        """
        if dataX_shell.size != dataY_shell.size:
            # Shells should have the same number of data points.
            err_str = f'Data1 shell size ({dataX_shell.size}) not compatible with' +\
                f'data2 shell size ({dataY_shell.size}).'
            raise ValueError(err_str)
        else:
            ###
            # Some cells only have values in one grid and not the other. 
            # We will subset these out.
            ###
            non_zero_ind = (dataX_shell != 0.0)*(dataY_shell != 0.0)

            # Eliminating cells that do not have a corresponding value.
            dataX_shell = dataX_shell[non_zero_ind]
            dataY_shell = dataY_shell[non_zero_ind]

        # Checking if there are weight values.
        if np.any(dataX_weights) and np.any(dataY_weights):
            # If there are weight values.
            dataX_weights = dataX_weights[non_zero_ind]
            dataY_weights = dataY_weights[non_zero_ind]
        else:
            # If there are no weights give equal value to each cell.
            dataX_weights = np.ones(dataX_shell.size)
            dataY_weights = np.ones(dataY_shell.size)
        
        # Calculating integral grid min and max values.
        x_min = np.min(dataX_shell)-std_fac*np.std(dataX_shell)
        x_max = np.max(dataX_shell)+std_fac*np.std(dataX_shell)
        y_max = np.max(dataY_shell)+std_fac*np.std(dataY_shell)
        y_min = np.min(dataY_shell)-std_fac*np.std(dataY_shell)

        # 1D grid coordinates.
        x = np.linspace(x_min,x_max,nside)
        y = np.linspace(y_min,y_max,nside)

        # Calculating the X and Y probability density values.
        px = KDEpy_1D_scaled(dataX_shell,weights=dataX_weights,x=x,bw_method=bw)
        py = KDEpy_1D_scaled(dataY_shell,weights=dataY_weights,x=y,bw_method=bw)

        # 2D meshgrid coordinates.
        xx,yy = np.meshgrid(x,y)

        # Initialising the data array for the 2D KDE.
        data_array = np.empty((len(dataX_shell),2))

        # Assigning the data values to the data array.
        data_array[:,0] = dataX_shell
        data_array[:,1] = dataY_shell

        # Defining the grid resolution.
        dx = np.abs(x[1]-x[0])
        dy = np.abs(y[1]-y[0])

        # Defining the 2D grid resolution.
        dx2D = np.abs(x[1]-x[0])
        dy2D = np.abs(y[1]-y[0])

        # Initialising a weights array.
        weights_array = np.empty((len(dataY_weights),2))

        # Assigning the weight values to the array.
        weights_array[:,0] = dataX_weights
        weights_array[:,1] = dataY_weights

        # Calculating the 2D probability density values.
        pxy = KDEpy_2D_scaled(data_array,weights_array,xx,yy,bw_method=bw)

        if plot_cond:
            file_path = f"/home/jaiden/Documents/Skewspec/output/MI_outputs/{dataX_shell.size}.png"
            # Plot the marginal distibution.
            scale = 0.75
            fig,axs = plt.subplots(1,figsize=(scale*12,scale*10))

            figaxs = (fig,axs)

            Plot_joint_marginal_dists(xx,yy,x,y,pxy,px,py,pxlab='data1',
                                      pylab='data2',figaxs=figaxs,lw=2.5,
                                      logcond=False,filename=file_path)

        # Calculate the differential entropy for the X, Y and XY distributions.
        diff_ent_X = MI_metric.diff_ent_1D(px,dx)
        diff_ent_Y = MI_metric.diff_ent_1D(py,dy)
        diff_ent_pxy = MI_metric.diff_ent_2D(pxy,dx2D,dy2D)

        # Calculating the mutual information.
        MI_temp = MI_metric.mutual_information(diff_ent_X,diff_ent_Y,diff_ent_pxy)

        return MI_temp

