#!/usr/bin/python

__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"

# Generic stuff:
import warnings
warnings.filterwarnings("ignore")

# Array stuff:
import numpy as np
warnings.simplefilter('ignore', np.RankWarning)

# Plotting stuff:
import matplotlib.pyplot as plt
import matplotlib

from dataclasses import dataclass

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

from spec import constants


def plot_spherical(k_r,Spec1D,figsize=(8,6),scale=1,xlim=None,ylim=None,
                   title=None,figaxs=None,xlabel=None,ylabel=None,step=True,
                   grid_cond=False,**kwargs):
    """
    Plot the 1D angular averaged power spectrum. If figaxs is provided allows for plotting
    more than one power spectrum.

    Parameters
    ----------
    k_r : numpy array, float
        1D vector of spherically radial k-modes.
    Spec1D : numpy array, float
        1D Power.
    figsize : tuple, default=(7.5,10.5)
        Size of the figure.
    scale : float, default=1
        Figure scaling factor.
    xlim : float, default=None
        X limits of the image pixels. In units of h Mpc^-1.
    ylim : float, default=None
        Y limits of the image pixels. In units of h Mpc^-1.
    title : str, default=None
        If given, the plot has a title.
    figaxs : tuple, default=None
        Tuple containing the matplotlib figure and axis objects. 
    xlabel : str, default=None
        xlabel string default is k [h Mpc^-1]
    ylabel : str, default=None
        ylabel string default is P(k) [mK^2 h^-3 Mpc^3]
    step : bool, default=True
        If True plot as a step plot.
    grid_cond : bool, default=False
        If True plot with a grid.
    
    Returns
    -------
    None
    """
    import matplotlib.ticker as ticker

    # Define a custom tick label formatting function
    def format_tick_label(x,pos):
        # Testing the dynamic range. If the dynamic range is too small
        # the asymmetric log plotting is fucked. In this case we perform our
        # own tick formatting.
        
        try:
            expo = int(np.log10(np.abs(x)))
            x_norm = x/10**(expo)

            return fr"{x_norm:.1f}$\times10^{{{expo}}}$"
        except OverflowError:

            return None

    if figaxs:
        # If figure and axis given.
        fig = figaxs[0]
        axs = figaxs[1]
    else:
        fig, axs = plt.subplots(1, figsize = figsize, dpi=75)

    if scale != 1:
        # If scale is not default, rescale the figure size.            
        figx = fig.get_figheight()*scale
        figy = fig.get_figwidth()*scale

        fig.set_figheight(figx)
        fig.set_figwidth(figy)

    # Determining the x and y labels.
    if xlabel:
        xlabel=xlabel
    else:
        xlabel=r'$k \,[\it{h}\rm{\,Mpc^{-1}}]$'

    if ylabel:
        ylabel=ylabel
    else:
        ylabel=r'$P(k) \, [\rm{mK^2}\,\it{h^{-3}}\,\rm{Mpc^3}]$'

    axs.set_xlabel(xlabel,fontsize=24*scale)
    axs.set_ylabel(ylabel,fontsize=24*scale)

    axs.tick_params(axis='x',labelsize=20*scale)
    axs.tick_params(axis='y',labelsize=20*scale)

    axs.set_xscale('log')
    if np.min(Spec1D) < 0:
        # If there are negative values set the scale to log symmetric.
        axs.set_yscale('asinh')
        
        thresh_limit = 0.5
        dynamic_range = (np.nanmax(Spec1D)/np.nanmin(Spec1D))
        
        if dynamic_range >= thresh_limit:
            # Testing the dynamic range. If the dynamic range is too small
            # the asymmetric log plotting is fucked. In this case we perform our
            # own tick formatting.
            print(dynamic_range)
            axs.set_yscale('linear')
            axs.yaxis.set_major_formatter(ticker.FuncFormatter(format_tick_label))
            axs.tick_params(axis='y',labelsize=12*scale)
    else:
        # Default is loglog axes scales. This can be manually changed
        # outside the function.
        axs.set_yscale('log')

    if step:
        # Default options is a step plot.
        axs.step(k_r,Spec1D,**kwargs)
    else:
        # Line plot is more useful for comparing Power spectra with different bin sizes.
        axs.plot(k_r,Spec1D,**kwargs)

    # Setting the x and y axis limits.
    if xlim:
        axs.set_xlim(xlim)
    if ylim:
        axs.set_ylim(ylim)


    # Changing the line widths.
    [x.set_linewidth(2.) for x in axs.spines.values()]

    axs.grid(grid_cond)

    if figaxs:
        if title:
            plt.savefig('{0}.png'.format(title))
        return axs
        
    else:
        plt.tight_layout()
    
        if title:
            plt.savefig('{0}.png'.format(title),bbox_inches='tight')
        else:
            plt.show()

def plot_cylindrical(Spec2D,kperp,kpar,figsize=(7.5,10.5),scale=1,cmap='Spectral_r',
    name=None,xlim=None,ylim=None,vmin=None,vmax=None,clab=None,lognorm=True,
    title=None,horizon_cond=False,Omega=0.076,z=6.8,verbose=False,**kwargs):
    """
    Plot the 2D cylindrically averaged Spectrum.

    Parameters
    ----------
    Spec2D : numpy array, float
        2D numpy array containing the power.
    kperp : numpy array, float
        1D vector of perpendicular k-modes.
    kpar : numpy array, float
        1D vector of parallel k-modes.
    figsize : tuple, default=(7.5,10.5)
        Size of the figure.
    scale : float, default=1
        Figure scaling factor.
    cmap : str, default='Spectral_r'
        Colour map to use. Default is Spectral_r, cividis, twighlight, or 
        viridis are also good options to consider.
    name : str, default=None
        If a name is given the figure is saved as name.png.
    xlim : float, default=None
        X limits of the image pixels. In units of h Mpc^-1.
    ylim : float, default=None
        Y limits of the image pixels. In units of h Mpc^-1.
    vmin : float, default=None
        Minimum value of the colorbar.
    vmax : float, default=None
        Maximum value of the colorbar.
    clab : str, default=None
        Colorbar label. If None, based on the sign of the minimum value,
        function will predict whether the plot is a power or skewspectrum.
    lognrom : bool, default=True
        Plot a lognorm colorbar. If the minimum value is negative will plot 
        the asinh symmetric log colorbar.
    title : str, default=None
        If given, the plot has a title.
    horizon_cond : bool, default=False
        If True plot the horizon and beam lines onto the 2D image. These calculations
        need to be double checked.
    Omega : float, default=0.076
        Solid angle of the MWA primary beam. Only used if horizon_cond=True.
    z : float, default=6.8
        Redshift of the spectrum. Only used if horizon_cond=True.
    verbose : bool, default=False
        If True print some additional details. Spectrum statistics.
    
    Returns
    -------
    None
    """

    fig, axs = plt.subplots(1, figsize = figsize, dpi=75, constrained_layout=True)

    if scale != 1:
        # If scale is not default, rescale the figure size.            
        figx = fig.get_figheight()*scale
        figy = fig.get_figwidth()*scale

        fig.set_figheight(figx)
        fig.set_figwidth(figy)

    if vmax:
        vmax=vmax
    else:
        pspec_max = np.max(np.log10(Spec2D[Spec2D > 0]))
        vmax = 10**pspec_max
    
    if vmin != None:
        # If vmin=0 this is considered false. So condition has to be different
        # to vmax.
        vmin=vmin
    else:
        pspec_min = (np.log10(np.abs(np.min(Spec2D))))

        if np.min(Spec2D) < 0:

            vmin = -1*10**pspec_min
        else:
            vmin = 10**pspec_min

    # If Trye plot lognormal colorbar. Default=True.
    if lognorm:
        if vmin < 0:
            # If less than zero make asihn symmetric lognorm colorbar.
            norm = matplotlib.colors.AsinhNorm(vmin=vmin,vmax=vmax,
                                           linear_width=0.1)
        else:
            norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)

    if verbose:
        print(f'Min = {np.min(Spec2D[Spec2D > 0]):5.3e}')
        print(f'Max = {np.max(Spec2D[Spec2D > 0].flatten()[0]):5.3e}')
    
    # Setting NaN values to a particular colour:
    cmap = matplotlib.cm.get_cmap(cmap)
    cmap.set_bad('lightgray',1.)

    # Creating the image object.
    im = axs.imshow(Spec2D,cmap=cmap,origin='lower',\
            extent=[np.min(kperp),np.max(kperp),np.min(kpar),np.max(kpar)],\
            norm=norm, aspect='auto',**kwargs)
    
    # Setting the colour bars:
    cb = fig.colorbar(im, ax=axs, aspect=40,pad=0.02, extend='both')

    if clab:
        # If label is provided in the function call.
        pass
    else:
        # If no label is provided in the function call.
        if vmin >= 0:
            # Default case where input spectrum is the power spectrum.
            # Power spectrum cannot be negative.
            clab = r'$P(k_\perp,k_{||}) \, [\rm{mK^2}\,\it{h^{-3}}\,\rm{Mpc^3}]$'
        else:
            # Default case where input spectrum is the skew spectrum.
            clab = r'$S(k_\perp,k_{||}) \, [\rm{mK^3}\,\it{h^{-3}}\,\rm{Mpc^3}]$'
    
    cb.set_label(label=clab,fontsize=20*scale)

    axs.set_xscale('log')
    axs.set_yscale('log')


    # If horizon condition is True then plot lines on the 2D plot.
    if horizon_cond:
        cosmo = constants.cosmo

        # Cosmological scaling parameter:
        h = cosmo.H(0).value/100 # Hubble parameter.
        E_z = cosmo.efunc(z) ## Scaling function, see (Hogg 2000)

        # Cosmological distances:
        Dm = cosmo.comoving_distance(z).value*h #[Mpc/h] Transverse co-moving distance.
        DH = (constants.c/1000)/100 # [Mpc/h] Hubble distance.

        # Full sky.
        grad_max = 0.5*np.pi*Dm*E_z/(DH*(1 + z)) # Morales et al (2012) horizon cosmology cut.
        # Primary beam FOV.
        #grad = np.sqrt(Omega)*Dm*E_z/(DH*(1 + z)) # Morales et al (2012) horizon cosmology cut.
        grad = 0.5*np.sqrt(Omega)*Dm*E_z/(DH*(1 + z)) # Morales et al (2012) horizon cosmology cut.

        line_width = 2.2

        # Illustrating the EoR window and horizon lines.
        axs.plot([0.1/grad_max,0.1],grad_max*np.array([0.1/grad_max,0.1]),lw=line_width,c='k')
        axs.plot([0.008,0.1/grad_max-0.0002839],[0.1,0.1],lw=line_width,c='k')
        axs.plot([0.1,0.1],[grad_max*(0.1+0.002),1.78],lw=line_width,c='k')
        axs.plot(kperp,grad*kperp,lw=line_width,ls='--',c='k')
    else:
        pass

    if xlim:
        axs.set_xlim(xlim)
    else:
        axs.set_xlim([0.008,np.max(kperp)])
        #axs.set_xlim([np.min(kperp),np.max(kperp)])
        
    if ylim:
        axs.set_ylim(ylim)
    else:
        axs.set_ylim([0.01,np.max(kpar)])

    axs.set_xlabel(r'$k_\perp \,[\it{h}\rm{\,Mpc^{-1}}]$',fontsize=20*scale)
    axs.set_ylabel(r'$k_{||}\,[\it{h}\rm{\,Mpc^{-1}}]$',fontsize=20*scale)

    # Setting the tick label fontsizes.
    axs.tick_params(axis='x', labelsize=18*scale)
    axs.tick_params(axis='y', labelsize=18*scale)
    cb.ax.tick_params(labelsize=18*scale)

    # Changing the line widths.
    [x.set_linewidth(2.) for x in axs.spines.values()]
    cb.outline.set_linewidth(2.)
    cb.outline.set_color('k')

    axs.grid(False)

    if title:
        plt.title(title,fontsize=20*scale)

    if name:
        plt.savefig('{0}.png'.format(name),bbox_inches='tight')
    else:
        plt.show()