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



# TODO:
# 1) Create the parent class and populate it with Osiris_spec_nu.py functions/
# 2) Create child powerspec and skewspec classes.
# 3) Think about how to include the KDE and MI stuff. They might need to be a
# separate module file.
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
    kb: float = 1380.649 #[Jy m^2 Hz K^-1] Boltzmann's constant.

    from astropy.cosmology import Planck18
    # In the case we don't provide a cosmology we have a prefered one.
    cosmo = Planck18

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
    calc_kr_grid(u_grid,v_grid,eta_vec,z,cosmo=None,kperp_cond=False)
        ...
    calc_field_of_view(sig_u)
        ...
    set_wedge_to_nan(self,kx_grid,ky_grid,kz_vec,kr_min,wedge_cut=None,
        horizon_cond=True)
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
                 weights_cube=None,cosmo=None):
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
        """
        self.cube = cube
        self.u_arr = u_arr # 2D u-grid, in units of wavelength.
        self.v_arr = v_arr # 2D u-grid, in units of wavelength.
        self.eta = eta # 1D vector of time values. 

        # Redefining the eta bins as per the CHIPS convention.
        Neta = len(self.eta)
        eta_nu = np.array([(float(i)-0.5)/(self.dnu*1e6) for i in range(Neta)])
        eta_nu[0] = eta_nu[1]/2
        self.eta = eta_nu

        # Defining the observation redshift.
        self.z = (constants.nu_21/self.nu_o) - 1
        self.nu_o = nu_o # [Hz]
        self.dnu = dnu # Bandwidth in [MHz].
        self.dnu_f = dnu_f # Fine channel width in [MHz].
        self.cosmo_factor = 1 # Depends on the spectrum. 

        if np.any(weights_cube):
            # Case for user inputted weigth cube.
            self.weights_cube = weights_cube

        else:
            print('Natural weighting case.')
            # Default weighting scheme is natural. Default set to not break older code.
            self.weights_cube = np.zeros(np.shape(cube))
            
            # Only cells with values are assigned weights.
            self.weights_cube[self.power_cube > 0.0] = 1.0
        
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
        DH = c/cosmo.H(0).value # approximately 3000 Mpc/h

        # k_||
        k_z = eta * (2*np.pi*nu_21*E_z)/(DH*(1 + z)**2) # [Mpc^-1 h]

        return k_z
    
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
        DH = c/cosmo.H(0).value # approximately 3000 Mpc/h

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
            print(f'Conversion factor = {conv_factor:5.3e} [mK^2 Hz^-2 Mpc^3 h^-3]')
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
        DH = c/cosmo.H(0).value # approximately 3000 Mpc/h

        # Bullshit magic number.
        # See Appendix page 20 Barry et al 2019 (FHD/epsilon) pipeline.
        deco_factor = 2 # Don't know if I need this.

        # Volume term.
        volume_term = (Dm**2 * DH *(1 + z)**2)/(nu_21 * E_z) # [sr^-1 Hz^-1 Mpc^3 h^-3]
        # Temperature term.
        temperature_term = (lam_o**6/(8*kb**3))

        # Converting a 1 Jy^2 source to K^3 Mpc^6 h^-6.
        conv_factor =  (dnu/(Omega_fov))*temperature_term*volume_term* 1e+9 # [mK^3 Mpc^3 h^-3]
        
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
            print(f'Conversion factor = {conv_factor:5.3e} [mK^3 Hz^-2 Mpc^3 h^-3]')
            print('==========================================================')
        else:
            pass
        
        return conv_factor
    
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
        DH = c/cosmo.H(0).value # approximately 3000 Mpc/h

        wedge_factor = Dm*E_z/(DH*(1 + z)) 

        return wedge_factor
    
    @staticmethod
    def calc_kr_grid(u_grid,v_grid,eta_vec,z,cosmo=None,kperp_cond=False):
        """
        Calculates the radial k-mode grid for an input u-grid, v-grid and eta-grid.
        It also performs the unit conversions from (u,v,eta) to (kx,ky,kz).
                
            Parameters
            ----------
            u_grid : numpy array
                2D grid of u-values in wavelengths.
            v_grid : numpy array
                2D grid of v-values in wavelengths.
            eta_vec : numpy array
                1D vector of eta values, in seconds.
            z : float
                Redshift value.
            cosmo : astropy object, default=None
                Astropy cosmology object, contains the Universe cosmology.
            kperp_cond : bool, default=False
                If True return the kx_grid, and ky_grid values.
            
            Returns
            -------
            kr_grid : numpy array
                3D numpy array containing the (kx,ky,kz) norm value for each voxel.
            kx_grid : numpy array, default=None
                Optional: if kperp_cond=True return the 2D kx_grid.
            ky_grid : numpy array, default=None
                Optional: if kperp_cond=True return the 2D ky_grid.
            kz_vec : numpy array, default=None
                Optional: if kperp_cond=True return the 1D kz_vec.
        """
        if cosmo == None:
            # If no cosmology provided use the defualt Plank2018 Cosmology.
            cosmo = constants.cosmo
        else:
            pass

        # Defining the kx, ky, and kz values from u,v and eta.
        kz_vec = polySpectra.eta2kz(eta_vec,z,cosmo) # [Mpc^-1 h]
        kx_grid = polySpectra.uv2kxky(u_grid,z,cosmo) # [Mpc^-1 h]
        ky_grid = polySpectra.uv2kxky(v_grid,z,cosmo) # [Mpc^-1 h]

        # Creating 3D k_r array.
        kr_grid = np.array([np.sqrt(kx_grid**2 + ky_grid**2 + kz**2) for kz in kz_vec]).T

        if kperp_cond:
            return kr_grid,kx_grid,ky_grid,kz_vec
        else:
            return kr_grid
    
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

    def set_wedge_to_nan(self,kx_grid,ky_grid,kz_vec,kr_min,wedge_cut=None,horizon_cond=True):
        """
        Use the kx, ky and kz grids to set the wedge values to zero. 
        
            Parameters
            ----------
            kx_grid : numpy array, default=None
                Optional: if kperp_cond=True return the 2D kx_grid.
            ky_grid : numpy array, default=None
                Optional: if kperp_cond=True return the 2D ky_grid.
            kz_vec : numpy array, default=None
                Optional: if kperp_cond=True return the 1D kz_vec.
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
        k_perp = np.sqrt(kx_grid**2 + ky_grid**2) # [Mpc^-1 h]
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
            np.array([np.logical_or(k_par < wedge_cut*k_perp, k_perp >= k_perp_min) for k_par in kz_vec]).T
        #wedge_ind_cube = np.array([k_perp >= k_perp_min for k_par in kz_vec]).T

        print(f'wedge_cut {wedge_cut:5.3f}')

        # Setting all k_par modes greater than some mode set to True.
        wedge_ind_cube[:,:,kz_vec < kr_min] = True

        # Setting the foreground wedge to zero.
        self.cube[wedge_ind_cube] = np.NaN
        self.weights_cube[wedge_ind_cube] = np.NaN
        self.kr_grid[wedge_ind_cube] = np.NaN
    
    def Spherical(self,func=np.average,wedge_cond=False,N_bins=60,sig=1.843,log_bin_cond=False,
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

            Nbins : int, default=60

            sig : float, default=1.843
                Width of a Gaussian primary beam in uv space. Units of Lambda. Beam is defined
                as e^x^2/sigma^2.

            kr_min : float, default=None

            kr_max : float, default=False

            log_bin_cond : bool, default=False

            horizon_cond : bool, default=True

            wedge_cut : float, default=None

            verbose : bool, default=False

            
            Returns
            -------
        """
        ### TODO
        # 1) Add an option to have user inputted kr_bins.
        # 2) Generalise the averaging process.

        # Calculating the field of view.
        self.Omega_fov = polySpectra.calc_field_of_view(sig)

        if wedge_cond:
            # If this is True we want to set all the voxels in the foreground wedge to be
            # NaN. This incluses their weight values as well.
            
            # Calculating the kr_grid.
            self.kr_grid,kx_grid,ky_grid,kz_vec = \
                polySpectra.calc_kr_grid(self.u_arr,self.v_arr,self.eta,self.z,self.cosmo)
            
            # Setting the wedge values to zero.
            polySpectra.set_wedge_to_nan(self,kx_grid,ky_grid,kz_vec,
                                        kr_min,wedge_cut=wedge_cut,horizon_cond=horizon_cond)
            del kx_grid,ky_grid

        else:
            # Default if we don't want to exclude the wedge.
            self.kr_grid = polySpectra.calc_kr_grid(self.u_arr,self.v_arr,self.eta,self.z,self.cosmo)

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

            if verbose: print(f'dk = {dk:5.3f}')

        if verbose:
            print(f'k_r_min = {kr_min:5.3f}')
            print(f'k_r_max = {kr_max:5.3f}')
            print(f'N_bins = {N_bins}')

        start0 = time.perf_counter()
        spec_avg_1D = np.zeros(N_bins)
        kr_vec = np.zeros(N_bins)

        # Indexing is faster in 1D arrays. If the arrays are filled.
        #self.kr_grid = self.kr_grid.ravel()
        #self.cube = self.cube.ravel()
        #self.weights_cube = self.weights_cube.ravel()
        
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

            try:
                # Some bins don't contain data, this will ensure the script doesn't fail. These bins will be set
                # to NaN.
                #spec_avg_1D[i] = np.average(self.cube[shell_ind],weights=self.weights_cube[shell_ind])
                spec_avg_1D[i] = func(self.cube[shell_ind],weights=self.weights_cube[shell_ind])
                
            except ZeroDivisionError and ValueError:
                # For bins that don't have data we will set these to NaN.
                spec_avg_1D[i] = np.nan
                
        end0 = time.perf_counter()
        
        print(f'1D spectrum calctime = {(end0-start0):5.3f} [s]')

        if verbose:
            print(f'Sigma = {sig:5.3f} [lambda]')
            print(f'Field of view {self.Omega_fov:5.4f} [Sr]')
            print(f"Conversion Jy squared into P(k) is {self.cosmo_factor:.5e}")

        self.spec_avg_1D = spec_avg_1D*self.cosmo_factor # [mK^2 Mpc^3 h^-3]
        self.k_r = kr_vec
    
    def Cylindrical(self,func=np.mean):
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
        """
        ### TODO
        # 1) Make this a static method, and generalise it. 

        #bin_width = 2.5 # [lambda]
        bin_width = 3.75 # [lambda]
        
        # Converting into cosmological values.
        dk_r = polySpectra.uv2kxky(bin_width,self.z,self.cosmo) # h Mpc^-1
        k_perp_max = polySpectra.uv2kxky(self.uvmax,self.z,self.cosmo) # h Mpc^-1
        
        # Defininf the number of bins.
        N_bins = int(k_perp_max/dk_r)
        
        # Specifying the radius vector:
        kr_bins = np.linspace(0,k_perp_max,N_bins + 1)

        # Problems with instrument sampling. Ommitting first bin.
        kr_bins = kr_bins[1:]

        # The u_arr and v_arr should be shifted. 
        r_uv = np.sqrt(self.u_arr**2 + self.v_arr**2)
        kr_uv_arr = polySpectra.uv2kxky(r_uv,self.z,self.cosmo)

        # Initialising the power spectrum and radius arrays.
        spec_avg_2D = np.zeros([len(self.eta),len(kr_bins)-1])
        kr_vec = np.zeros(len(kr_bins)-1)

        # Averaging the power in annular rings for each eta slice.
        for i in range(len(self.eta)):
            for j in range(len(kr_bins)-1):
                
                # Assigning the radius values. Needed for plotting purposes.
                kr_vec[j] = ((kr_bins[j+1] + kr_bins[j])/2.0)

                # Creating annunuls of boolean values.
                temp_ind = np.logical_and(kr_uv_arr >= kr_bins[j], kr_uv_arr <= kr_bins[j+1])

                # Weighted averaging of annuli values.
                try:
                    # Some bins don't contain data, this will ensure the script doesn't fail. These bins will be set
                    # to NaN.
                    #spec_avg_2D[i,j] = np.average(self.cube[temp_ind,i],weights=self.weights_cube[temp_ind,i]) 
                    spec_avg_2D[i,j] = func(self.cube[temp_ind,i],weights=self.weights_cube[temp_ind,i]) 
                    
                except ZeroDivisionError:
                    # For bins that don't have data we will set these to NaN.
                    spec_avg_2D[i,j] = np.nan

        # Assigning the power.
        self.spec_avg_2D = spec_avg_2D*self.cosmo_factor # [mK^3 Mpc^3 h^-3]

        # Assigning the perpendicular and parallel components of the power spectrum.
        self.kperp = kr_vec
        self.kpar = polySpectra.eta2kz(self.eta,self.z,self.cosmo)

class powerSpec(polySpectra):
    """
    Test child class.
    """

    def __init__(self,cube,u_arr,v_arr,eta):
        super().__init__(cube,u_arr,v_arr,eta)

        # Overriding attributes to suite the power-spectrum.
        self.cube = np.conjugate(cube)*cube # [Jy^2 Hz^2]
        self.cosmo_factor = (1/(self.dnu_f*1e+6)**2)*polySpectra.Power2Tb(self.dnu*1e+6,self.dnu_f*1e+6,
                                            self.nu_o,self.z,self.cosmo,self.Omega_fov)




class skewSpec(polySpectra):
    """
    Test child class.
    """

    def __init__(self,cube,cubesqd,u_arr,v_arr,eta):
        super().__init__(cube,u_arr,v_arr,eta)

        self.cube = cubesqd*cube

class kdeSpec(polySpectra):
    """
    Test child class.
    """

    def __init__(self,cube,u_arr,v_arr,eta):
        super().__init__(cube,u_arr,v_arr,eta)

        self.cube = cube

class miSpec(polySpectra):
    """
    Class to calculate the mutual information (MI) of two input arrays.
    """

    def __init__(self,cube,cube2,u_arr,v_arr,eta):
        super().__init__(cube,u_arr,v_arr,eta)

        self.cube1 = cube
        self.cube2 = cube2