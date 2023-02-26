#!/usr/bin/python

__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"

# Generic stuff:
import time
import warnings
warnings.filterwarnings("ignore")

# Array stuff:
import numpy as np
warnings.simplefilter('ignore', np.RankWarning)

# Plotting stuff:
import matplotlib.pyplot as plt

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


def plot_spherical(k_r,Power1D,figsize=(8,6),xlim=None,ylim=None,title=None,figaxs=None,\
    xlabel=None,ylabel=None,step=True,scale=1,**kwargs):
    """
    Plot the 1D angular averaged power spectrum. If figaxs is provided allows for plotting
    more than one power spectrum.

        Parameters
        ----------
        k_r : numpy array, float
            1D vector of spherically radial k-modes.
        Power1D : numpy array, float
            1D Power.
        
        Returns
        -------
        None
    """

    if figaxs:
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

    plt.loglog()

    if step:
        # Default options is a step plot.
        axs.step(k_r,Power1D,**kwargs)
    else:
        # Line plot is more useful for comparing Power spectra with different bin sizes.
        axs.plot(k_r,Power1D,**kwargs)


    if xlim:
        axs.set_xlim(xlim)
    if ylim:
        axs.set_ylim(ylim)

    if xlabel:
        axs.set_xlabel(xlabel,fontsize=24)
    else:
        axs.set_xlabel(r'$k \,[\it{h}\rm{\,Mpc^{-1}}]$',fontsize=24)

    if ylabel:
        axs.set_ylabel(ylabel,fontsize=24)
    else:
        axs.set_ylabel(r'$P(k) \, [\rm{mK^2}\,\it{h^{-3}}\,\rm{Mpc^3}]$',fontsize=24)

    axs.tick_params(axis='x',labelsize=20)
    axs.tick_params(axis='y',labelsize=20)

    axs.grid(False)

    if figaxs:
        if title:
            plt.savefig('{0}.png'.format(title))
        return axs
        
    else:
        plt.tight_layout()
    
        if title:
            plt.savefig('{0}.png'.format(title))
        else:
            plt.show()


def plot_cylindrical(Power2D,kperp,kpar,figsize=(7.5,10.5),cmap='viridis',
    name=None,xlim=None,ylim=None,vmin=None,vmax=None,clab=None,lognorm=True,
    title=None,horizon_cond=False,scale=1,**kwargs):

    """
    Plot the 2D cylindrically averaged power spectrum.

        Parameters
        ----------
        Power2D : numpy array, float
            2D numpy array containing the power.
        kperp : numpy array, float
            1D vector of perpendicular k-modes.
        kpar : numpy array, float
            1D vector of parallel k-modes.
        
        Returns
        -------
        None
    """

    ### TODO: Change the FoV calculations. We have a FoV we can use that.
    #fov = 0.076
    
    # Performing a rudimentary fov calculation:
    # Not 100% sure these are correct, but they create cuts similar to Trott et al 2020.
    sig = 4 # lambda
    sig = 2 # lambda
    FWHM = 2*np.sqrt(2*np.log(2))/sig
    fov = FWHM**2 # rad**2


    #print(fov)

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
        pspec_max = np.max(np.log10(Power2D[Power2D > 0]))
        vmax = 10**pspec_max
    
    if vmin != None:
        # If vmin=0 this is considered false. So condition has to be different
        # to vmax.
        vmin=vmin
    else:
        pspec_min = np.min(np.log10(Power2D[Power2D > 0]))

        vmin = 10**pspec_min

    if lognorm:
        norm = matplotlib.colors.LogNorm()
        #norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = None

    print('Min = %5.3e' % np.min(Power2D[Power2D > 0]))
    print('Max = %5.3e' % np.max(Power2D[Power2D > 0].flatten()[0]))
    
    # Setting NaN values to a particular colour:
    #cmap = matplotlib.cm.viridis
    cmap = matplotlib.cm.get_cmap("Spectral_r")
    cmap.set_bad('lightgray',1.)

    im = axs.imshow(Power2D,cmap=cmap,origin='lower',\
            extent=[np.min(kperp),np.max(kperp),np.min(kpar),np.max(kpar)],\
            norm=norm,vmin=vmin,vmax=vmax, aspect='auto',**kwargs)
    

    # Setting the colour bars:
    cb = fig.colorbar(im, ax=axs, fraction=0.04, pad=0.002, extend='both')

    if clab:
        cb.set_label(label=clab,fontsize=20)
    else:
        cb.set_label(label=r'$\rm{P(k_\perp,k_{||})} \, [\rm{mK^2}\,\it{h^{-3}}\,\rm{Mpc^3}]$',fontsize=20)
    
    axs.set_xscale('log')
    axs.set_yscale('log')

    ####
    from astropy.cosmology import Planck18

    # Cosmological scaling parameter:
    z = 7.14
    h = Planck18.H(0).value/100 # Hubble parameter.
    E_z = Planck18.efunc(z) ## Scaling function, see (Hogg 2000)

    # Cosmological distances:
    Dm = Planck18.comoving_distance(z).value*h #[Mpc/h] Transverse co-moving distance.
    DH = 3000 # [Mpc/h] Hubble distance.

    # Full sky.
    grad_max = 0.5*np.pi*Dm*E_z/(DH*(1 + z)) # Morales et al (2012) horizon cosmology cut.
    # Primary beam FOV.
    #grad = np.sqrt(fov)*Dm*E_z/(DH*(1 + z)) # Morales et al (2012) horizon cosmology cut.
    grad = 0.5*np.sqrt(fov)*Dm*E_z/(DH*(1 + z)) # Morales et al (2012) horizon cosmology cut.

    if horizon_cond:
        
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

    axs.set_xlabel(r'$k_\perp \,[\it{h}\rm{\,Mpc^{-1}}]$',fontsize=20)
    axs.set_ylabel(r'$k_{||}\,[\it{h}\rm{\,Mpc^{-1}}]$',fontsize=20)

    # Setting the tick label fontsizes.
    axs.tick_params(axis='x', labelsize=18)
    axs.tick_params(axis='y', labelsize=18)
    cb.ax.tick_params(labelsize=18)

    axs.grid(False)

    if title:
        plt.title(title,fontsize=20)
    else:
        pass

    if name:
        plt.savefig('{0}.png'.format(name))
    else:
        plt.show()