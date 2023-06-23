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

from KDEspec import MI_metric

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

# TODO:
# 4) Generalise the plotting functions.

def progress_bar(index,Niter,percent_cond=False):
    """
    Convert u or v into k_x or k_y, k_z as per Morales et al. (2004).
    Uses the Plank 2018 cosmology as default. 

    Can convert r = sqrt(u^2 + v^2) to k_perp. Same formula.
            
    Parameters
    ----------
    index : int
        For loop index.
    Niter : int
        Total number of iterations.
    percent_cond : bool, default=False
        If True display progress in percentages.
    
    Returns
    -------
    None
    """
    import sys
    # Progress bar:
    if (index+1) % 10 == 0:
        if percent_cond:
            percent = float((index+1)/Niter)*100
            stout_str = f'\rPercent processed {percent:5.3f} %'
        else:
            stout_str = f"\rBins processed: {(index+1)}/{Niter}"
        
        sys.stdout.write(stout_str)
        sys.stdout.flush()
    elif (index+1) == Niter:
        # Last iteration.
        if percent_cond:
            sys.stdout.write(f"\rBins processed: 100 %\n")
        else:
            sys.stdout.write(f"\rBins processed: {Niter}/{Niter}\n")
        sys.stdout.flush()
    else:
        pass

@dataclass
class constants:
    """
    Data class containing all the fundamental physical constants used in the
    polySpectra base class. 
    """

    # Constants
    c: float = 299792458.0 #[m/s]
    lam_21: float = 0.21 #[m]
    nu_21: float = c/lam_21 #[Hz]
    kb: float = 1380.649 #[Jy m^2 K^-1] Boltzmann's constant.

    # Default cosmology.
    from astropy.cosmology import LambdaCDM
    omega_matter = 0.31
    omega_baryon = 0.048
    omega_lambda = 0.69
    hubble = 68
    ##Create the astropy cosmology
    cosmo = LambdaCDM(H0=hubble,Om0=omega_matter,Ode0=omega_lambda,Ob0=omega_baryon)

class polySpectra:
    """
    Parent class for the poly spec objects. Provides a template for power spectrum
    skew spectrum, KDE power spectrum and Mutual information spectrum calculations. 

    Attributes
    ----------
    cube : numpy array
        Input data cube, either power, skew or other.
    u_arr : numpy array
        Input u grid in units of lambda. 2D numpy array.
    v_arr : numpy array
        Input v grid in units of lambda. 2D numpy array.
    eta : numpy array
        Input eta grid in units of time.
    nu_o : float
        Observing frequency at the centre of the band [Hz].
    dnu : float, default=30.72e6
        Observing bandwidth [Hz]. Should be 30.72MHz.
    dnu_f : float, default=80e3
        Fine channel width [Hz]. Should be 80kHz by default.
    weights_cube : numpy array, default=None
        Weights of the data points. If None assumed to be naturally weighted.
    cosmo : astropy object, default=None
        Astropy Cosmology object, default used is Plank2018.


    Methods
    -------
    uv2kxky(u,z,cosmo=None)
        ...
    eta2kz(eta,z,cosmo=None)
        ...
    Power2Tb(dnu,dnu_f,nu_o,z,cosmo,Omega_fov,verbose=True):
        ...
    Skew2Tb(dnu,dnu_f,nu_o,z,cosmo,Omega_fov,verbose=True)
        ...
    wedge_factor(z,cosmo=None)
        ...
    calc_kr_grid(u_grid,v_grid,z,eta_vec=None,cosmo=None,return_kxyz=False)
        ...
    calc_field_of_view(sig_u)
        ...
    set_wedge_to_nan(self,kx_grid,ky_grid,kz_vec,kr_min,wedge_cut=None,
        horizon_cond=True)
        ...
    avgWrapper(self,shell_ind)
        ...
    avgSpherical(self,wedge_cond=False,N_bins=60,sig=1.843,log_bin_cond=False,
                  kr_min=None,kr_max=None,horizon_cond=True,wedge_cut=None,verbose=False
        ...
    avgCylindrical(self)
        ...
    Spherical(self,func=np.average,wedge_cond=False,N_bins=60,sig=1.843,
        log_bin_cond=False,kr_min=None,kr_max=None,horizon_cond=True,wedge_cut=None,
        verbose=False)
        ...
    Cylindrical(self,func=np.mean)
        ...
    """
    constants = constants
    def __init__(self,cube,u_arr,v_arr,eta,nu_o,dnu=30.72e6,dnu_f=80e3,
                 weights_cube=None,cosmo=None,uvmax=300,sig=1.843,ravel_cond=False):
        """
        Constructs the spectrum object.

        Parameters
        ----------
        cube : numpy array
            Input data cube, either power, skew or other.
        u_arr : numpy array
            Input u grid in units of lambda. 2D numpy array.
        v_arr : numpy array
            Input v grid in units of lambda. 2D numpy array.
        eta : numpy array
            Input eta grid in units of time.
        nu_o : float
            Observing frequency at the centre of the band [Hz].
        dnu : float, default=30.72e6
            Observing bandwidth [Hz]. Should be 30.72MHz.
        dnu_f : float, default=80e3
            Fine channel width [Hz]. Should be 80kHz by default.
        weights_cube : numpy array, default=None
            Weights of the data points. If None assumed to be naturally weighted.
        cosmo : astropy object, default=None
            Astropy Cosmology object, default used is Plank2018.
        uvmax : float, defualt=300
            uv cutoff in wavelengths.
        """
        self.cube = cube
        self.u_arr = u_arr # 2D u-grid, in units of wavelength.
        self.v_arr = v_arr # 2D u-grid, in units of wavelength.
        self.eta = eta # 1D vector of time values. 
        self.uvmax = uvmax
        
        # Overide this in the child methods.
        self.ravel_cond = ravel_cond

        # Defining the observation redshift.
        self.nu_o = nu_o # [Hz]
        self.z = (constants.nu_21/self.nu_o) - 1
        self.dnu = dnu # Bandwidth in [MHz].
        self.dnu_f = dnu_f # Fine channel width in [MHz].
        self.cosmo_factor = 1 # Depends on the spectrum. 

        # Redefining the eta bins as per the CHIPS convention.
        Neta = len(self.eta)
        #eta_nu = np.array([(float(i)-0.5)/(self.dnu*1e6) for i in range(Neta)])
        eta_nu = np.array([(float(i)-0.5)/(self.dnu) for i in range(Neta)])
        eta_nu[0] = eta_nu[1]/2
        self.eta = eta_nu
        
        self.Omega_fov = polySpectra.calc_field_of_view(sig)

        if np.any(weights_cube):
            self.weights_cube = weights_cube
        else:
            # Might change this to be specific to each of the different spectra
            # Sub classes

            # Setting the weights cube.
            self.weights_cube = np.zeros(np.shape(cube))
                
            # Only cells with values are assigned weights.
            self.weights_cube[self.cube > 0.0] = 1.0

        if cosmo != None:
            # User inputted cosmology.
            print('Not using Plank2018 Cosmology.')
            self.cosmo = cosmo
        else:
            # Default is the Plank18 cosmology.
            print('Using Plank2018 Cosmology.')
            self.cosmo = constants.cosmo
    

    @staticmethod
    def uv2kxky(u,z,cosmo=None):
        """
        Convert u or v into k_x or k_y, k_z as per Morales et al. (2004).
        Uses the Plank 2018 cosmology as default. 

        Can convert r = sqrt(u^2 + v^2) to k_perp. Same formula.
                
        Parameters
        ----------
        u_arr : numpy array, float
            NDarray of u or v values. Should be in wavelengths.
        z : float
            Redshift at the central frequency of the band.
        cosmo : astropy object, default=None
            Astropy cosmology object. Default is Plank2018 cosmology.
        
        Returns
        -------
        k_vec : numpy array, float
            NDarray of k-mode values. Should be in units of h*Mpc^-1. 
        """
        if cosmo == None:
            # If no cosmology provided use the defualt Plank2018 Cosmology.
            cosmo = constants.cosmo
        else:
            pass

        # Cosmological scaling parameter:
        h = cosmo.H(0).value/100 # Hubble parameter.
    
        # Cosmological distances:
        Dm = cosmo.comoving_distance(z).value*h #[Mpc/h] Transverse co-moving distance.

        # Converting u to k
        k_vec = u * (2*np.pi/Dm) # [Mpc^-1 h]

        return k_vec

    @staticmethod
    def eta2kz(eta,z,cosmo=None):
        """
        Convert eta into k_z as per Morales et al. (2004).
        Uses the Plank 2018 cosmology as default.
                
        Parameters
        ----------
        eta : numpy array, float
            1Darray of eta values. 
        z : float
            Redshift at the central frequency of the band.
        cosmo : astropy object, default=None
            Astropy cosmology object. Default is Plank2018 cosmology.
        
        Returns
        -------
        k_z : numpy array, float
            1Darray of kz values. Should be in units of h*Mpc^-1.
        """
        if cosmo == None:
            # If no cosmology provided use the defualt Plank2018 Cosmology.
            cosmo = constants.cosmo
        else:
            pass

        # Constant:
        c = constants.c #[m/s]
        nu_21 = constants.nu_21 # [Hz]

        # Cosmological scaling parameter:
        E_z = cosmo.efunc(z) ## Scaling function, see (Hogg 2000)

        # Cosmological distances:
        #DH = 3000 # [Mpc/h] Hubble distance.
        DH = (c/1000)/100 # approximately 3000 Mpc/h

        # k_||
        k_z = eta * (2*np.pi*nu_21*E_z)/(DH*(1 + z)**2) # [Mpc^-1 h]

        return k_z
    
    @staticmethod
    def spec2Del_spec(k_r,spec):
        """
        Converts a spectrum from unit*h^-3 Mpc^3 to unit. Here unit is usually
        mK^2 or mK^3. 

        Parameters
        ----------
        k_r : float, numpy array
            k_r vector [Mpc^-1] or [h Mpc^-1]
        spec : float, numpy aray
            1D spectra vector in units of Mpc^3 or h^-3 Mpc^3.
        
        Returns
        -------
        Del_spec : float, numpy array
            1D unitless vector.
        """

        # Check that both arrays have the same dimension.
        if len(k_r) != len(spec):

            err_msg = f'len(k_r) = {len(k_r)} != len(spec) = {len(spec)}.'
            raise ValueError(err_msg)

        # Calc the Del_spectrum.
        Del_spec = spec*((k_r)**3)/(2*np.pi**2)

        return Del_spec


    @staticmethod
    def Power2Tb(dnu,dnu_f,nu_o,z,cosmo,Omega_fov,verbose=True):
        """
        Calculate the conversion factor from Jy^2 Hz^2 to mK^2 Mpc^3 h^-3.

        Parameters
        ----------
        dnu : float
            Bandwidth [Hz].
        dnu_f : float
            Fine channel width [Hz].
        nu_o : float
            Observing frequency at the centre of the band [Hz].
        z : float
            Redshift of the observing frequency.
        cosmo : astropy object
            Astropy Cosmology object, default used is Plank2018.
        
        Returns
        -------
        conv_factor
        """
        # Constants
        c = constants.c #[m/s]
        nu_21 = constants.nu_21 #[Hz]
        kb = constants.kb # [Jy m^2 Hz K^-1] Boltzmann's constant.

        # Constants.
        lam_o = c/nu_o #[m]
        N_chans = dnu/dnu_f

        # Cosmological scaling parameter:
        h = cosmo.H(0).value/100
        E_z = cosmo.efunc(z)

        # Cosmological distances:
        Dm = cosmo.comoving_distance(z).value*h #[Mpc/h]
        #DH = 3000 # [Mpc/h] Hubble distance.
        DH = (c/1000)/100 # approximately 3000 Mpc/h

        # Volume term.
        volume_term = (Dm**2 * DH *(1 + z)**2)/(nu_21 * E_z) # [sr^-1 Hz^-1 Mpc^3 h^-3]
        # Temperature term, converts from Jy^2 Sr^2 to K^2
        temperature_term = (lam_o**4/(4*kb**2))

        # Bullshit magic number.
        # See Appendix page 20 Barry et al 2019 (FHD/epsilon) pipeline.
        deco_factor = 2 # Don't know if I need this.

        # Converting a 1 Jy^2 source to mK^2 Mpc^3 h^-3.
        conv_factor =  deco_factor*temperature_term*volume_term*(dnu/Omega_fov)*1e+6 # [mK^2 Mpc^3 h^-3]

        if verbose:
            print('==========================================================')
            print('Cosmology values')
            print('==========================================================')
            print(f'Bandwidth = {dnu:5.1f} [Hz]')
            print(f'DM = {Dm:5.3f} [Mpc/h]')
            print(f'DH = {DH:5.3f} [Mpc/h]')
            print(f'h = {h:5.3f}')
            print(f'FoV = {Omega_fov:5.4f} [Sr]')
            print(f'z = {z:5.3f}')
            print(f'E(z) = {E_z:5.3f}')
            print(f'Decoherence factor = {deco_factor}')
            print(f'N_chans = {N_chans}')
            print(f'Observed wavelength = {lam_o:5.3f} [m]')
            print(f'Fine channel width = {dnu_f:5.3e} [Hz]')
            print(f'Volume term = {volume_term:5.3f} [sr^-1 Hz^-1 Mpc^3 h^-3]')
            print(f'Conversion factor = {conv_factor:5.3e} [mK^2 Mpc^3 h^-3]')
            print('==========================================================')
        else:
            pass
        
        return conv_factor

    @staticmethod
    def Skew2Tb(dnu,dnu_f,nu_o,z,cosmo,Omega_fov,verbose=True):
        """
        Calculate the conversion factor from Jy^2 Hz^2 to mK^2 Mpc^3 h^-3.

        Parameters
        ----------
        dnu : float
            Bandwidth [Hz].
        dnu_f : float
            Fine channel width [Hz].
        nu_o : float
            Observing frequency at the centre of the band [Hz].
        z : float
            Redshift of the observing frequency.
        cosmo : astropy object
            Astropy Cosmology object, default used is Plank2018.
        
        Returns
        -------
        conv_factor
        """
        # Constants
        c = constants.c #[m/s]
        nu_21 = constants.nu_21 #[Hz]
        kb = constants.kb # [Jy m^2 Hz K^-1] Boltzmann's constant.

        # Constants.
        lam_o = c/nu_o #[m]
        # Future versions we will have the field of view as an input. 
        N_chans = dnu/dnu_f

        # Cosmological scaling parameter:
        h = cosmo.H(0).value/100
        E_z = cosmo.efunc(z)

        # Cosmological distances:
        Dm = cosmo.comoving_distance(z).value*h #[Mpc/h]
        #DH = 3000 # [Mpc/h] Hubble distance.
        DH = (c/1000)/100 # approximately 3000 Mpc/h

        # Bullshit magic number.
        # See Appendix page 20 Barry et al 2019 (FHD/epsilon) pipeline.
        deco_factor = 2 # Don't know if I need this.

        # Volume term.
        volume_term = (Dm**2 * DH *(1 + z)**2)/(nu_21 * E_z) # [sr^-1 Hz^-1 Mpc^3 h^-3]
        # Temperature term.
        temperature_term = (lam_o**6/(8*kb**3))

        # Converting a 1 Jy^2 source to K^3 Mpc^6 h^-6.
        conv_factor =  deco_factor*(dnu/(Omega_fov))*temperature_term*volume_term* 1e+9 # [mK^3 Mpc^3 h^-3]
        
        if verbose:
            print('==========================================================')
            print('Cosmology values')
            print('==========================================================')
            print(f'Bandwidth = {dnu:5.1f} [Hz]')
            print(f'Nu21 = {nu_21:5.1f} [Hz]')
            print(f'DM = {Dm:5.3f} [Mpc/h]')
            print(f'DH = {DH:5.3f} [Mpc/h]')
            print(f'h = {h:5.3f}')
            print(f'FoV = {Omega_fov:5.4f} [Sr]')
            print(f'z = {z:5.3f}')
            print(f'E(z) = {E_z:5.3f}')
            print(f'Decoherence factor = {deco_factor}')
            print(f'N_chans = {N_chans}')
            print(f'Observed wavelength = {lam_o:5.3f} [m]')
            print(f'Fine channel width = {dnu_f:5.3e} [Hz]')
            print(f'Volume term = {volume_term:5.3f} [sr^-1 Hz^-1 Mpc^3 h^-3]')
            print(f'Conversion factor = {conv_factor:5.3e} [mK^3 Mpc^3 h^-3]')
            print('==========================================================')
        else:
            pass
        
        return conv_factor
    
    @staticmethod
    def calc_field_of_view(sig_u):
        """
        Calculates the field of view in steradians. The primary beam in this case is 
        assumed to be Gaussian and defined in (l,m) space. The input given is sig_u 
        which is the width of the FT of the beam. We can use this to determine the beam
        in (l,m) space. 

        The width is defined for the following Gaussian function G = G0 exp(-x^2/sigma^2).
        Therefore sigma_prime = sigma/root(2). Keep this in mind. In future I may revert 
        back to a more sensible definition.
        
        Parameters
        ----------
        sig_u : float, or numpy array
            Value(s) of grid kernel widths.
        
        
        Returns
        -------
        Omega_fov : float, or numpy array
            Field of view value(s) in Sr.
        """
        # Estimating the sky width.
        sigma_b = 1/(np.pi*sig_u)
        sigma_b_prime = sigma_b/np.sqrt(2)

        # Gaussian beam width.
        w = 2*sigma_b_prime

        # Using this to calculate the FoV.
        Omega_fov = np.pi*w**2

        # FoV assuming FWHM of G = G0 exp(-0.5 x^2 / sig^2) kernel definition.
        #Omega_fov = (2*np.log(2))/(np.pi*(sig)**2) # Omega = 2 ln(2) / (pi* sig_grid^2) [Sr]

        return Omega_fov

    @staticmethod
    def wedge_factor(z,cosmo=None):
        """
        Nicholes horizon cosmology cut.
                
        Parameters
        ----------
        z : float
            Redshift.
        cosmo : astropy object, default=None
            Astropy cosmology object, default is None. If None use Plank18 cosmology.
        
        Returns
        -------
        wedge_factor : float
            k|| > wedge_factor * k_perp cut.
        """
        if cosmo == None:
            # If no cosmology provided use the defualt Plank2018 Cosmology.
            cosmo = constants.cosmo
        else:
            pass

        # Cosmological scaling parameter:
        h = cosmo.H(0).value/100 # Hubble parameter.
        E_z = cosmo.efunc(z) ## Scaling function, see (Hogg 2000)
        c = constants.c # speed of light m/s

        # Cosmological distances:
        Dm = cosmo.comoving_distance(z).value*h #[Mpc/h] Transverse co-moving distance.
        #DH = 3000 # [Mpc/h] Hubble distance.
        DH = (c/1000)/100 # approximately 3000 Mpc/h

        wedge_factor = Dm*E_z/(DH*(1 + z)) 

        return wedge_factor
    
    @staticmethod
    def calc_kr_grid(u_grid,v_grid,z,eta_vec=None,cosmo=None):
        """
        Calculates the radial k-mode grid for an input u-grid, v-grid and eta-grid.
        If eta is a single value then only calculate the grid for a single slice.
        It also performs the unit conversions from (u,v,eta) to (kx,ky,kz).
                
        Parameters
        ----------
        u_grid : numpy array
            2D grid of u-values in wavelengths.
        v_grid : numpy array
            2D grid of v-values in wavelengths.
        z : float
            Redshift value.
        eta_vec : numpy array, default=None
            1D vector of eta values, in seconds. If None 2D Grid calculation.
        cosmo : astropy object, default=None
            Astropy cosmology object, contains the Universe cosmology.
        
        Returns
        -------
        kr_grid : numpy array
            3D or 2D numpy array containing the (kx,ky) or (kx,ky,kz) norm value for each voxel.
        """
        if cosmo == None:
            # If no cosmology provided use the defualt Plank2018 Cosmology.
            cosmo = constants.cosmo
        else:
            pass

        # Defining the kx, ky, and kz values from u,v and eta.
        kx_grid = polySpectra.uv2kxky(u_grid,z,cosmo) # [Mpc^-1 h]
        ky_grid = polySpectra.uv2kxky(v_grid,z,cosmo) # [Mpc^-1 h]

        if np.any(eta_vec):
            # 3D case.
            # kz can be a float or a vector.
            kz_vec = polySpectra.eta2kz(eta_vec,z,cosmo) # [Mpc^-1 h]

            # Creating 3D k_r array.
            kr_grid = np.array([np.sqrt(kx_grid**2 + ky_grid**2 + kz**2) for kz in kz_vec]).T
        else:
            # 2D case.
            # Creating 2D k_perp array
            kr_grid = np.sqrt(kx_grid**2 + ky_grid**2)

        return kr_grid

    def set_wedge_to_nan(self,kr_min,wedge_cut=None,horizon_cond=True):
        """
        Use the kx, ky and to set the wedge values to zero. 
        
        Parameters
        ----------
        kr_min : float
            Minimum radius value.
        wedge_cut : float, default=None
            User can input their own wedge cut value.
        horizon_cond : bool, default=True
            If True use the horizon as the cut instead of the primary beam.
        
        
        Returns
        -------
        None
        """
        # Calculating the k_perp array.
        kz_vec = polySpectra.eta2kz(self.eta,self.z,self.cosmo) # [Mpc^-1 h]
        k_perp = polySpectra.calc_kr_grid(self.u_arr,self.v_arr,self.z,cosmo=self.cosmo) # [Mpc^-1 h]
        # Specifying a minimum k_perp.
        k_perp_min = 0.1 # [Mpc^-1 h]

        if wedge_cut:
            # Option to manually input a wedge cut value.
            pass
        else:
            # Default is to calculate the horizon or beam grad.
            if horizon_cond:
                grad = 0.5*np.pi # Horizon cut gradient.
            else:
                grad = 0.5*np.sqrt(self.Omega_fov) # Beam FoV cut.

            # Nicholes horizon cosmology cut.
            wedge_cut = grad*polySpectra.wedge_factor(self.z,self.cosmo) 

        # Calculating the wedge mask array.
        wedge_ind_cube = \
            np.array([np.logical_or(k_par < wedge_cut*k_perp, k_perp >= k_perp_min) for k_par in self.eta]).T
        #wedge_ind_cube = np.array([k_perp >= k_perp_min for k_par in kz_vec]).T

        print(f'wedge_cut {wedge_cut:5.3f}')

        # Setting all k_par modes greater than some mode set to True.
        wedge_ind_cube[:,:,kz_vec < kr_min] = True

        # Setting the foreground wedge to zero.
        self.cube[wedge_ind_cube] = np.NaN
        self.weights_cube[wedge_ind_cube] = np.NaN
        self.kr_grid[wedge_ind_cube] = np.NaN
    

    def avgWrapper(self,shell_ind):
        """
        Wrapper for calculating the MI. Calculates both the 1D and 2D array values.

        Parameters
        ----------
        self : object
            Power object contains u and v arrays, as well as the observation redshift.
        shell_ind : numpy array
            Numpy array of boolean values. This is the shell index, either a spherical or
            circular shell. If ind is not None this is spherical, circular otherwise.
        
        Returns
        -------
        avg_shell_power : float
            Average shell power.
        """
        try:
            avg_shell_power = np.average(self.cube[shell_ind],
                                         weights=self.weights_cube[shell_ind])
        except ZeroDivisionError:
            avg_shell_power = np.nan

        return avg_shell_power
    
    def avgSpherical(self,wedge_cond=False,N_bins=60,sig=1.843,log_bin_cond=False,
                  kr_min=None,kr_max=None,horizon_cond=True,wedge_cut=None,verbose=False):
        """
        Wrapper for calculating the spherical average.

        Parameters
        ----------

        wedge_cond : bool, default=False
            If True wedge values set to NaN.
        Nbins : int, default=60
            Number of bines to average.
        sig : float, default=1.843
            Width of a Gaussian primary beam in uv space. Units of Lambda. Beam is defined
            as e^x^2/sigma^2.
        kr_min : float, default=None
            Min kr value.
        kr_max : float, default=False
            Max kr value.
        log_bin_cond : bool, default=False
            If True bins are loglinear spaced.
        horizon_cond : bool, default=True
            If True use the horizon as the cut instead of the primary beam.
        wedge_cut : float, default=None
            User can input their own wedge cut value.
        verbose : bool, default=False
            If True, print additional details.
        """
        polySpectra.Spherical(self,func=polySpectra.avgWrapper,
                            wedge_cond=wedge_cond,N_bins=N_bins,sig=sig,
                            log_bin_cond=log_bin_cond,kr_min=kr_min,kr_max=kr_max,
                            horizon_cond=horizon_cond,wedge_cut=wedge_cut,verbose=verbose)

    
    def avgCylindrical(self):
        """
        Wrapper for calculating the spherical average.
        """
        polySpectra.Cylindrical(self,func=polySpectra.avgWrapper)


    def Spherical(self,func,wedge_cond=False,N_bins=60,sig=1.843,log_bin_cond=False,
                  kr_min=None,kr_max=None,horizon_cond=True,wedge_cut=None,verbose=False):
        """
        Calculates the 1D spherically averaged poly spectra using the input object.
                
        Parameters
        ----------
        self : object
            Power object contains u and v arrays, as well as the observation redshift.
        func : function, default=np.average
            Input Averaging function, KDE, MI, mean...
        wedge_cond : bool, default=False
            If True wedge values set to NaN.
        Nbins : int, default=60
            Number of bines to average.
        sig : float, default=1.843
            Width of a Gaussian primary beam in uv space. Units of Lambda. Beam is defined
            as e^x^2/sigma^2.
        kr_min : float, default=None
            Min kr value.
        kr_max : float, default=False
            Max kr value.
        log_bin_cond : bool, default=False
            If True bins are loglinear spaced.
        horizon_cond : bool, default=True
            If True use the horizon as the cut instead of the primary beam.
        wedge_cut : float, default=None
            User can input their own wedge cut value.
        verbose : bool, default=False
            If True, print additional details.

        
        Returns
        -------
        self
        """
        ### TODO
        # 1) Add an option to have user inputted kr_bins.

        # Calculating the field of view.
        self.Omega_fov = polySpectra.calc_field_of_view(sig)

        if wedge_cond:
            # If this is True we want to set all the voxels in the foreground wedge to be
            # NaN. This incluses their weight values as well.
            polySpectra.set_wedge_to_nan(self,kr_min,wedge_cut=wedge_cut,horizon_cond=horizon_cond)
        
        # Calculating the kr_grid.
        self.kr_grid = polySpectra.calc_kr_grid(self.u_arr,self.v_arr,self.z,self.eta,self.cosmo)

        if kr_min:
            # User can manually input a kr min.
            kr_min = float(kr_min)
        else:
            kr_min = np.nanmin(self.kr_grid[self.kr_grid > 0.0])

        if kr_max:
            # User can manually input a kr max.
            kr_max = float(kr_max)
        else:
            kr_max = np.nanmax(self.kr_grid)

        if log_bin_cond:
            # Logarithmically spaced bins, creates uniform bin widths in log space plots.
            # Binning conditions are different for log bins.

            # Log-linear bins.
            log_kr_min = np.log10(kr_min)
            log_kr_max = np.log10(kr_max)
            
            # Increments.
            dlog_k = (log_kr_max - log_kr_min)/(N_bins + 1)

            k_r_bins = np.logspace(log_kr_min - dlog_k/2,log_kr_max + dlog_k/2,N_bins + 1)
            if verbose: print(f'dlog_k = {dlog_k:5.3e}')

        else:
            # Increments.
            dk = (kr_max - kr_min)/(N_bins + 1)
            k_r_bins = np.linspace(kr_min,kr_max,N_bins + 1)

            if verbose: 
                print(f'dk = {dk:5.3f}')

        if verbose:
            print(f'k_r_min = {kr_min:5.3f}')
            print(f'k_r_max = {kr_max:5.3f}')
            print(f'N_bins = {N_bins}')

        start0 = time.perf_counter()
        spec_avg_1D = np.zeros(N_bins)
        kr_vec = np.zeros(N_bins)

        # Indexing is faster in 1D arrays. If the arrays are filled.
        if self.ravel_cond:
            # If the data array is 1D then flatten the coordinate grid.
            self.kr_grid = self.kr_grid.ravel()
        
        for i in range(len(k_r_bins)-1):

            # Show a progress bar.
            progress_bar(i,N_bins,percent_cond=True)

            # Calculating the radius:
            if log_bin_cond:
                kr_vec[i] = 10**(0.5*(np.log10(k_r_bins[i+1]) + np.log10(k_r_bins[i])))
            else:
                kr_vec[i] = ((k_r_bins[i+1] + k_r_bins[i])/2.0)

            # Defining the shell array index:
            shell_ind = np.logical_and(self.kr_grid >= k_r_bins[i], self.kr_grid <= k_r_bins[i+1])

            spec_avg_1D[i] = func(self,shell_ind)
                
        end0 = time.perf_counter()
        
        print(f'\n1D spectrum calctime = {(end0-start0):5.3f} [s]')

        if verbose:
            print(f'Sigma = {sig:5.3f} [lambda]')
            print(f'Field of view {self.Omega_fov:5.4f} [Sr]')
            print(f"Conversion Jy squared into P(k) is {self.cosmo_factor:.5e}")

        self.spec_avg_1D = spec_avg_1D*self.cosmo_factor # [mK^2 Mpc^3 h^-3]
        self.k_r = kr_vec
    
    def Cylindrical(self,func):
        """
        Calculates the 2D cylindrical power spectra using the input Power object.
                
        Parameters
        ----------
        self : object
            Power object contains u and v arrays, as well as the observation redshift.
        func : function, default=np.average
            Input Averaging function, KDE, MI, mean...

        
        Returns
        -------
        self
        """
        start0 = time.perf_counter()

        bin_width = 3.75 # [lambda]
        
        # Converting into cosmological values.
        dk_r = polySpectra.uv2kxky(bin_width,self.z,self.cosmo) # h Mpc^-1
        # Calculating kperp max.
        k_perp_max = polySpectra.uv2kxky(self.uvmax,self.z,self.cosmo) # h Mpc^-1
        
        # Defininf the number of bins.
        N_bins = int(k_perp_max/dk_r)
        
        # Specifying the radius vector:
        kr_bins = np.linspace(0,k_perp_max,N_bins + 1)

        # Problems with instrument sampling. Ommitting first bin.
        kr_bins = kr_bins[1:]

        # Calculating the kperp array. 
        kperp_arr = polySpectra.calc_kr_grid(self.u_arr,self.v_arr,
                                             self.z,cosmo=self.cosmo) # [Mpc^-1 h]

        # Initialising the power spectrum and radius arrays.
        spec_avg_2D = np.zeros([len(self.eta),len(kr_bins)-1])
        kr_vec = np.zeros(len(kr_bins)-1)
        
        # If things are flattened then you need some fancy indexing.
        gridyy,gridxx = np.indices(kperp_arr.shape)

        # We only need to calculate the kperp bins once.
        # They are the same for each eta bin.
        kperp_index_list = []
        for j in range(len(kr_bins)-1):
            temp_ind = (kperp_arr >= kr_bins[j]) & (kperp_arr <= kr_bins[j+1])
            kperp_index_list.append(temp_ind)

        # Averaging the power in annular rings for each eta slice.
        for i in range(len(self.eta)):

            progress_bar(i,len(self.eta),percent_cond=True)

            for j,temp_ind in enumerate(kperp_index_list):
                
                # Assigning the radius values. Needed for plotting purposes.
                kr_vec[j] = ((kr_bins[j+1] + kr_bins[j])/2.0)

                # Getting annulus index values.
                gridyy_temp = gridyy[temp_ind]
                gridxx_temp = gridxx[temp_ind]

                N_samp_temp = int(gridxx_temp.size)

                # Creating the 3D index array.
                arr_ind = np.array([gridyy_temp,gridxx_temp,np.ones(N_samp_temp,dtype=int)*i])
                if self.ravel_cond:
                
                    arr_ind = np.ravel_multi_index(arr_ind, kperp_arr.shape + (len(self.eta),))

                # Weighted averaging of annuli values.
                try:
                    # Some bins don't contain data, this will ensure the script doesn't fail. These bins will be set
                    # to NaN.
                    spec_avg_2D[i,j] = func(self,arr_ind)
                    
                except ZeroDivisionError and ValueError:
                    # For bins that don't have data we will set these to NaN.
                    spec_avg_2D[i,j] = np.nan

        # Assigning the power.
        self.spec_avg_2D = spec_avg_2D*self.cosmo_factor # [mK^3 Mpc^3 h^-3]

        end0 = time.perf_counter()
        
        print(f'\n2D spectrum calctime = {(end0-start0):5.3f} [s]')

        # Assigning the perpendicular and parallel components of the power spectrum.
        self.kperp = kr_vec
        self.kpar = polySpectra.eta2kz(self.eta,self.z,self.cosmo)

class powerSpec(polySpectra):
    """
    Class to calculate the spherical and cylindrical power spectrum.
    """

    def __init__(self,cube,u_arr,v_arr,eta,nu_o,dnu=30.72e6,dnu_f=80e3,
                         weights_cube=None,cosmo=None,sig=1.843,ravel_cond=False):
        super().__init__(cube,u_arr,v_arr,eta,nu_o,dnu=dnu,dnu_f=dnu_f,
                         weights_cube=weights_cube,cosmo=cosmo,sig=sig,
                         ravel_cond=ravel_cond)
        

        # Overriding attributes to suite the power-spectrum.
        self.cube = (np.conjugate(cube)*cube).real # [Jy^2 Hz^2]
        self.cosmo_factor = (1/(self.dnu_f)**2)*polySpectra.Power2Tb(self.dnu,self.dnu_f,
                                            self.nu_o,self.z,self.cosmo,self.Omega_fov)
        
        print('Power cube DC sum.')
        print(np.sum(self.cube[:,:,0]))

        if np.any(weights_cube):
            # Case for user inputted weigth cube.
            self.weights_cube = weights_cube
        
        if self.ravel_cond:
            self.cube = self.cube.ravel()
            self.weights_cube = self.weights_cube.ravel()
        
        del cube,weights_cube

class skewSpec(polySpectra):
    """
    Class to calculate the spherical and cylindrical skew spectrum.
    """

    def __init__(self,cube,cubesqd,u_arr,v_arr,eta,nu_o,dnu=30.72e6,dnu_f=80e3,
                         weights_cube=None,cosmo=None,sig=1.843,ravel_cond=False):
        super().__init__(cube,u_arr,v_arr,eta,nu_o,dnu=dnu,dnu_f=dnu_f,
                         weights_cube=weights_cube,cosmo=cosmo,sig=sig,
                         ravel_cond=ravel_cond)

        # We only need the real values.
        self.cube = (cubesqd*np.conjugate(cube)).real # [Jy^3 Hz^2]
        self.cosmo_factor = (1/(self.dnu_f)**2)*polySpectra.Skew2Tb(self.dnu,self.dnu_f,
                                            self.nu_o,self.z,self.cosmo,self.Omega_fov)

        print('Skew cube DC sum.')
        print(np.sum(self.cube[:,:,0]))
        print(np.sum(cubesqd[:,:,0]),np.sum(cube[:,:,0]))
        del cube,cubesqd

        if np.any(weights_cube):
            # Case for user inputted weigth cube.
            self.weights_cube = weights_cube
        
        if self.ravel_cond:
            self.cube = self.cube.ravel()
            self.weights_cube = self.weights_cube.ravel()

class miSpec(polySpectra,MI_metric):
    """
    Class to calculate the mutual information (MI) of two input arrays.
    """

    def __init__(self,cube,cubeY,u_arr,v_arr,eta,nu_o,cubeX_weights=None,cubeY_weights=None,
                 dnu=30.72e6,dnu_f=80e3,cosmo=None,ravel_cond=True,sig=1.843):
        super().__init__(cube,u_arr,v_arr,eta,nu_o,dnu=dnu,dnu_f=dnu_f,
                         cosmo=cosmo,sig=sig)
        
        # If True then the input arrays are flattened. This can speed up the calculation
        # process.
        if ravel_cond:
            self.cubeX = cube.ravel()
            self.cubeY = cubeY.ravel()
            self.ravel_cond = ravel_cond
        else:
            self.cubeX = cube
            self.cubeY = cubeY
            self.ravel_cond = ravel_cond

        del cube,cubeY
        del self.cube, self.weights_cube

        if np.any(cubeX_weights):
            # If weights provided they are assigned.
            if ravel_cond:
                self.cubeX_weights = cubeX_weights.ravel()
            else:
                self.cubeX_weights = cubeX_weights
        else:
            # No weights given assign weights.
            self.cubeX_weights = np.zeros(self.cubeX.shape)
            self.cubeX_weights[self.cubeX.shape != 0] = 1
        
        if np.any(cubeY_weights):
            # If weights provided they are assigned.
            if ravel_cond:
                self.cubeY_weights = cubeY_weights.ravel()
            else:
                self.cubeY_weights = cubeY_weights
        else:
            # No weights given assign weights.
            self.cubeY_weights = np.zeros(self.cubeY.shape)
            self.cubeY_weights[self.cubeY.shape != 0] = 1

        
    def MI_shell_wrapper(self,shell_ind):
        """
        Wrapper for calculating the MI. Calculates both the 1D and 2D array values.

        Parameters
        ----------
        self : object
            Power object contains u and v arrays, as well as the observation redshift.
        shell_ind : numpy array
            Numpy array of boolean values. This is the shell index, either a spherical or
            circular shell. If ind is not None this is spherical, circular otherwise.
        
        Returns
        -------
        MI : float
            Output mutual information value.
        """
        plot_cond = False
        try:
            MI = MI_metric.calc_KDE_MI(self.cubeX[shell_ind],self.cubeY[shell_ind],
                                dataX_weights=self.cubeX_weights[shell_ind],
                                dataY_weights=self.cubeY_weights[shell_ind],
                                plot_cond=plot_cond)
        except ZeroDivisionError or ValueError:
            MI = np.nan
    
        return MI
    
    def MI_spherical(self,wedge_cond=False,N_bins=60,sig=1.843,log_bin_cond=False,
                  kr_min=None,kr_max=None,horizon_cond=True,wedge_cut=None,verbose=False):
        """
        Wrapper for calculating the spherical MI.
        """
        miSpec.Spherical(self,func=miSpec.MI_shell_wrapper,wedge_cond=wedge_cond,N_bins=N_bins,
                         sig=sig,log_bin_cond=log_bin_cond,kr_min=kr_min,kr_max=kr_max,
                         horizon_cond=horizon_cond,wedge_cut=wedge_cut,verbose=verbose)

        self.MI_1D = self.spec_avg_1D
        del self.spec_avg_1D
    
    def MI_cylindrical(self):
        """
        Wrapper for calculating the spherical MI.
        """
        miSpec.Cylindrical(self,func=miSpec.MI_shell_wrapper)

        self.MI_2D = self.spec_avg_2D
        del self.spec_avg_2D
    

class kdeSpec(polySpectra):
    """
    Class to calculate the power spectrum using kernel density estimators.
    """

    def __init__(self,cube,u_arr,v_arr,eta,nu_o,dnu=30.72e6,dnu_f=80e3,
                         weights_cube=None,cosmo=None,sig=1.843):
        super().__init__(cube,u_arr,v_arr,eta,nu_o,dnu=dnu,dnu_f=dnu_f,
                         weights_cube=weights_cube,cosmo=cosmo,sig=sig)

        self.cube = cube