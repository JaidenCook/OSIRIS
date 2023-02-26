#!/usr/bin/python

__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@student.curtin.edu"

# Generic stuff:
#%matplotlib notebook
import time
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

# Array stuff:
import numpy as np
warnings.simplefilter('ignore', np.RankWarning)

# Plotting stuff:
import matplotlib.pyplot as plt
import matplotlib

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

# Parser options:
from optparse import OptionParser

# Scipy stuff:
import scipy
from scipy.fft import ifftn,fftn,fftfreq,fftshift,ifftshift
from scipy import stats
import scipy.optimize as opt

def Mean_pixel_solid_angle(N=1291,freq=136*1.28e+6,L=2,M=2,delays=[0]*16):
    """
    Estimates the variance of a temperature brightness field by integrating the power spectrum.
            
    Parameters
        ----------
        N : int
            Image array size. Default is 1291.
        freq : float
            Centre band frequency.
        L : float
            Size in direction cosine. Default is 2.
        M : float
            Size in direction cosine. Default is 2.
        delays : float, list
            MWA primary beam delays, default is zenith pointing.
        N : integer
            Image size in pixel units. Default is 1291.
        
        Returns
        -------
        dOmega_mean : float
            Expected field of fiew pixel solid angle in Steradians.
    """
    freq = 136*1.28e+6 #[Hz]

    # Initialising the l and m vectors.
    l_vec = np.linspace(-L/2,L/2,N)
    m_vec = np.linspace(-M/2,M/2,N)

    dl = np.abs(l_vec[1]-l_vec[0])
    dm = np.abs(m_vec[1]-m_vec[0])
    dA = dl*dm

    # Creating the grid:
    l_arr, m_arr = np.meshgrid(l_vec,m_vec)
    m_arr = m_arr[::-1,:] # Should probably fix this issue with the sky-model class.

    # Creating a radius array for masking purposes:
    r_arr = np.sqrt(l_arr**2 + m_arr**2)

    # Creating an index array, we want all pixels less than or equal to r = 1:
    ind_arr = r_arr <= 1.0

    # Here we want to create a new alt and az array that is the same size as l_arr and m_arr:
    Alt_arr = np.zeros(np.shape(l_arr))
    Az_arr = np.zeros(np.shape(l_arr))

    # Now we want to determine the Altitude and Azimuth, but only in the region where r <= 1. Outside this region is 
    # beyond the boundary of the horizon.
    Alt_arr[ind_arr] = np.arccos(r_arr[ind_arr]) # Alt = arccos([l^2 + m^2]^(1/2))
    Az_arr[ind_arr] = np.arctan2(l_arr[ind_arr],m_arr[ind_arr]) + np.pi #arctan2() returns [-pi,pi] we want [0,2pi].

    import mwa_hyperbeam

    # Creating MWA beam object.
    beam = mwa_hyperbeam.FEEBeam()

    beam_arr = np.zeros(np.shape(Alt_arr))

    # Hyperbeam options.
    amps = [1.0] * 16
    norm_to_zenith = True

    Zen_temp = np.pi/2 - Alt_arr

    temp_jones = beam.calc_jones_array(Az_arr[ind_arr],Zen_temp[ind_arr],freq,delays,amps,norm_to_zenith)

    xx_temp = temp_jones[:,0]*np.conjugate(temp_jones[:,0]) + temp_jones[:,1]*np.conjugate(temp_jones[:,1])
    yy_temp = temp_jones[:,2]*np.conjugate(temp_jones[:,2]) + temp_jones[:,3]*np.conjugate(temp_jones[:,3])

    beam_arr[ind_arr] = np.abs(xx_temp + yy_temp)/2
    beam_arr[r_arr >= 1.0] = np.nan

    sky_solid_angle_arr = np.zeros((N,N))

    N_ant = 16.0
    ind_arr = beam_arr >= 1/N_ant

    sky_solid_angle_arr[ind_arr] = dA/(np.sin(Alt_arr[ind_arr]))

    dOmega_mean = np.mean(sky_solid_angle_arr[ind_arr])

    return dOmega_mean

def Var_powerspec(k_r,Power1D,lam_o=1.71,N=1291,unit='Janksy'):
    """
    Estimates the variance of a temperature brightness field by integrating the power spectrum.
            
    Parameters
        ----------
        k_r : float, array
            Array of k values, has units Mpc^-1 h
        Power1D : float, array
            Array of 1D power spectrum values. Has units mK^2 Mpc^3 h-^3.
        lam_o : float
            Observation frame wavelength of cosmological 21cm signal. Has units of m.
            Default value is 1.71, redshift of 7.14.
        N : integer
            Image size in pixel units. Default is 1291.
        
        Returns
        -------
        Var : float
            Power spectrum variance estimate. 
    """
    ### 
    # Add this to the Osiris.Power_spec 
    ###
    from scipy import integrate

    lam_o = 1.64
    freq = 3e8/lam_o

    # Constants. ## Fix this.
    kb = 1380.648 # [Jy m^2 Hz K^-1] Boltzmann's constant.

    # Expected pixel solid angle.
    # Calculated from beam Beam_arr > 1/N_ant (Mort et al 2016)
    # Parameters given so you can see the inputs.
    #dOmega = Mean_pixel_solid_angle(N=1291,freq=136*1.28e+6,L=2,M=2,delays=[0]*16)
    dOmega = Mean_pixel_solid_angle(N=1291,freq=freq,L=2,M=2,delays=[0]*16)

    #dOmega = 2.401e-6

    print('dOmega = %5.3e [Sr]' % dOmega)

    # Bin size for the numerical intergration. 
    # Problems might occur if the bins are uniform in log-space.
    dk = np.diff(k_r)[0]
    print('dk = %5.3f' % dk)

    if unit=='Janksy':
        # Converting from mK^2 Mpc^3 h^-3 to Jy^2 Mpc^3 h^-3
        conv_factor = (1e-6)*((4*kb**2)/lam_o**4)*dOmega**2

        Power1D = Power1D*conv_factor
    else:
        pass

    # Normalisation factor. 
    norm_factor = 1/(2*np.pi)**3
    Var_PS = norm_factor*(4*np.pi)*integrate.trapezoid(k_r**2 * Power1D,x=k_r,dx=dk)

    # Calculating the trapezoidal error. Estimate seems low.
    N = len(k_r)
    f = norm_factor*(4*np.pi)*k_r**2 * Power1D
    M = np.abs(np.max(np.diff(f,n=2))) # maximum absolute difference.

    # Error Estimate.
    Err_up = (M*dk**3)/(12*N**2)

    if unit=='Janksy':
        print('Variance = %5.5f [Jy^2]' % Var_PS)
        print('Variance = %5.5e [Jy^2]' % Var_PS)
        print('Trapezoidal Error Est = %5.3e [Jy^2]' % Err_up)
    else:
        print('Variance = %5.5e [mK^2]' % Var_PS)
        print('Trapezoidal Error Est = %5.3e [mK^2]' % Err_up)

    return Var_PS

class Power_spec:
    """
    This class defines the different power spectrums. It allows for the calculation of the cylindrical and the
    angular averaged power spectrum. These are also referred to as the 2D and 1D power spectrums.

    ...

    Attributes
    ----------
    fine : numpy array
        ...


    Methods
    -------
    Power2Tb(freqs=""):
        ...
    uv_to_kxky(freqs=""):
        ...
    eta_to_kz(freqs=""):
        ...
    wedge_factor()
        ...
    Spherical()
        ...
    Cylindrical()
        ...
    plot_spherical()
        ...
    plot_cylindrical()
    """

    # Constants
    c = 299792458.0/1000 #[km/s]
    nu_21 = (1000*c)/(0.21) #[Hz]
    kb = 1380.649 # [Jy m^2 Hz K^-1] Boltzmann's constant.
    
    def __init__(self,Four_sky_cube,eta,u_arr,v_arr,nu_o,dnu,dnu_f,weights_cube=None,nu_21=nu_21,cosmo=None):
        
        # Attributes will include data cube, eta, 
        # For determining the 2D bins.
        #self.uvmax = np.max(u_arr)
        self.uvmax = 300

        #self.data = Four_sky_cube
        self.power_cube = np.conjugate(Four_sky_cube)*Four_sky_cube # [Jy^2 Hz^2]
        
        # We are calculating the one sided power spectrum. Which is a factor of 2 greater than the double sided,
        # for all modes except the DC mode at eta = 0.
        #self.power_cube[:,:,1:] *= np.sqrt(2)

        if np.any(weights_cube):
            # Case for user inputted weigth cube.
            self.weights_cube = weights_cube

        else:
            print('Natural weighting case.')
            # Default weighting scheme is natural. Default set to not break older code.
            self.weights_cube = np.zeros(np.shape(Four_sky_cube))
            
            # Only cells with values are assigned weights.
            self.weights_cube[self.power_cube > 0.0] = 1.0

            #print(np.sum(self.weights_cube))
        
        self.eta = eta # [Hz^-1]
        self.u_arr = u_arr # Should have units of wavelengths.
        self.v_arr = v_arr # Should have units of wavelengths.
        self.nu_o = nu_o # [Hz]
        self.z = (nu_21/self.nu_o) - 1
        self.dnu = dnu # Bandwidth in [MHz].
        self.dnu_f = dnu_f # Fine channel width in [MHz].
        print('Redshift = %5.2f' % self.z)

        if cosmo != None:
            # User inputted cosmology.
            print('Using non-standard cosmology.')
            self.cosmo = cosmo
        else:
            # Default is the Plank18 cosmology.
            from astropy.cosmology import Planck18

            self.cosmo = Planck18


        # Save memory.
        del Four_sky_cube
    
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
        from scipy import signal

        # Constants
        c = 299792458.0 #[m/s]
        nu_21 = c/(0.21106114) #[Hz]
        kb = 1380.649 # [Jy m^2 Hz K^-1] Boltzmann's constant.

        # Constants.
        #lam_21 = 1000*c/nu_21 #[m]
        lam_o = c/nu_o #[m]
        #fov = 0.076 # [sr] field of view. Approximate.
        N_chans = dnu/dnu_f

        # Calculating the volume correction factor:
        window = signal.blackmanharris(int(N_chans))
        #Ceff = np.sum(window)/N_chans
        Ceff = np.sum(window)/N_chans

        # Cosmological scaling parameter:
        h = cosmo.H(0).value/100
        E_z = cosmo.efunc(z)

        # Cosmological distances:
        Dm = cosmo.comoving_distance(z).value*h #[Mpc/h]
        DH = 3000 # [Mpc/h] Hubble distance.

        # Volume term.
        co_vol = (Dm**2 * DH *(1 + z)**2)/(nu_21 * E_z) # [sr^-1 Hz^-1 Mpc^3 h^-3]

        # Bullshit magic number.
        # See Appendix page 20 Barry et al 2019 (FHD/epsilon) pipeline.
        deco_factor = 2 # Don't know if I need this.

        # Converting a 1 Jy^2 source to mK^2 Mpc^3 h^-3.
        #conv_factor = deco_factor * (N_chans**2) * (lam_o**4/(4*kb**2)) * (1/(Omega_fov*dnu)) * co_vol * 1e+6 # [mK^2 Mpc^3 h^-3]
        #conv_factor =  (1/N_chans)* deco_factor * (lam_o**4/(4*kb**2)) * (dnu/Omega_fov) * co_vol * 1e+6 # [mK^2 Mpc^3 h^-3]
        conv_factor =  deco_factor * (lam_o**4/(4*kb**2)) * (dnu/Omega_fov) * co_vol * 1e+6 # [mK^2 Mpc^3 h^-3]

        if verbose:
            print('==========================================================')
            print('Cosmology values')
            print('==========================================================')
            print('Bandwidth = %5.3f [Hz]' % dnu)
            print('DM = %5.3f [Mpc/h]' % Dm)
            print('DH = %5.3f [Mpc/h]' % DH)
            print('h = %5.3f' % h)
            print('FoV = %5.4f [Sr]' % Omega_fov)
            print('z = %5.3f' % z)
            print('E(z) = %5.3f' % E_z)
            print('Decoherence factor = %s' % deco_factor)
            print('N_chans = %s' % int(N_chans))
            print('Observed wavelength = %5.3f [m]' % lam_o)
            print('Fine channel width = %5.3e [Hz]' % dnu_f)
            print('Volume term = %5.3f [sr^-1 Hz^-1 Mpc^3 h^-3]' % co_vol)
            print('Conversion factor = %5.3f [mK^2 Mpc^3 sr^-2 h^-3]' % conv_factor)
            print('==========================================================')
            
            
        else:
            pass
        
        return conv_factor

    @staticmethod
    def uv_to_kxky(u,z,cosmo):
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
            
            Returns
            -------
            k_vec : numpy array, float
                NDarray of k-mode values. Should be in units of h*Mpc^-1. 
        """

        # Cosmological scaling parameter:
        h = cosmo.H(0).value/100 # Hubble parameter.
    
        # Cosmological distances:
        Dm = cosmo.comoving_distance(z).value*h #[Mpc/h] Transverse co-moving distance.

        # Converting u to k
        k_vec = u * (2*np.pi/Dm) # [Mpc^-1 h]

        return k_vec

    @staticmethod
    def eta_to_kz(eta,z,cosmo):
        """
        Convert eta into k_z as per Morales et al. (2004).
        Uses the Plank 2018 cosmology as default.
                
        Parameters
            ----------
            eta : numpy array, float
                1Darray of eta values. 
            z : float
                Redshift at the central frequency of the band.
            
            Returns
            -------
            k_z : numpy array, float
                1Darray of kz values. Should be in units of h*Mpc^-1.
        """

        # Constant:
        c = 299792458.0 #[m/s]
        nu_21 = c/(0.21106114) #[Hz]

        # Cosmological scaling parameter:
        E_z = cosmo.efunc(z) ## Scaling function, see (Hogg 2000)

        # Cosmological distances:
        DH = 3000 # [Mpc/h] Hubble distance.

        # k_||
        k_z = eta * (2*np.pi*nu_21*E_z)/(DH*(1 + z)**2) # [Mpc^-1 h]

        return k_z

    @staticmethod
    def wedge_factor(z,cosmo):
        """
        Nicholes horizon cosmology cut.
                
        Parameters
            ----------
            z : float
                Redshift.
            cosmo : Astropy Object
                Astropy cosmology object, default is None. If None use Plank18 cosmology.
            
            Returns
            -------
            wedge_factor : float
                k|| > wedge_factor * k_perp cut.
        """

        # Cosmological scaling parameter:
        h = cosmo.H(0).value/100 # Hubble parameter.
        E_z = cosmo.efunc(z) ## Scaling function, see (Hogg 2000)

        # Cosmological distances:
        Dm = cosmo.comoving_distance(z).value*h #[Mpc/h] Transverse co-moving distance.
        DH = 3000 # [Mpc/h] Hubble distance.

        wedge_factor = Dm*E_z/(DH*(1 + z)) 

        return wedge_factor

    def Spherical(self,wedge_cond=False,N_bins=60,log_bin_cond=False,kr_min=None,kr_max=None,
                horizon_cond=True,wedge_cut=None,sig=2.5):
        """
        Calculates the 1D spherical power spectra using the input Power object.
                
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

        # Defining the kx, ky, and kz values from u,v and eta.
        k_z = Power_spec.eta_to_kz(self.eta,self.z,self.cosmo) # [Mpc^-1 h]
        k_x = Power_spec.uv_to_kxky(self.u_arr,self.z,self.cosmo) # [Mpc^-1 h]
        k_y = Power_spec.uv_to_kxky(self.v_arr,self.z,self.cosmo) # [Mpc^-1 h]

        # Creating 3D k_r array.
        self.k_r_arr = np.array([np.sqrt(k_x**2 + k_y**2 + kz**2) for kz in k_z]).T

        if wedge_cond:
            # Condition for ignoring the wedge contribution to the power spectrum.
            
            if sig:
                # If the grid kernel size (sig) provided calculate the FoV.
                FWHM = 2*np.sqrt(2*np.log(2))/sig
                fov = FWHM**2 # [rad**2]

            else:
                fov = 0.076 # [rad**2]

            # Creating the k_perp array.
            k_perp = np.sqrt(k_x**2 + k_y**2) # [Mpc^-1 h]
            # Specifying a minimum k_perp.
            k_perp_min = 0.1 # [Mpc^-1 h]
            
            if horizon_cond:
                grad = 0.5*np.pi # Horizon cut gradient.
            else:
                grad = 0.5*np.sqrt(fov) # Beam FoV cut.

            if wedge_cut:
                # Option to manually input a wedge cut value.
                wedge_cut = wedge_cut
            else:
                # Default is to calculate the horizon or beam grad.
                wedge_cut = grad*Power_spec.wedge_factor(self.z,self.cosmo) # Nicholes horizon cosmology cut.
            
            print('wedge_cut %5.3f' % wedge_cut)

            # Calculating the wedge mask array.
            wedge_ind_cube = np.array([np.logical_or(k_par < wedge_cut*k_perp, k_perp >= k_perp_min) for k_par in k_z]).T
            #wedge_ind_cube = np.array([k_par < wedge_cut*k_perp for k_par in k_z]).T

            if kr_min:
                # User can manually input a kr min.
                kr_min = float(kr_min)
            else:
                ## Testing
                kr_min = 0.1

            # Setting all k_par modes greater than some mode set to True.
            wedge_ind_cube[:,:,k_z < kr_min] = True
            
            ## 
            # Setting the wedge to zero.
            self.power_cube[wedge_ind_cube] = np.NaN
            self.weights_cube[wedge_ind_cube] = np.NaN
            self.k_r_arr[wedge_ind_cube] = np.NaN

            # Calculating kr_max.
            kr_max = np.nanmax(self.k_r_arr)

        else:

            if kr_min:
                # User can manually input a kr min.
                kr_min = float(kr_min)
            else:
                kr_min = np.nanmin(self.k_r_arr[self.k_r_arr > 0.0])

            if kr_max:
                # User can manually input a kr max.
                kr_max = float(kr_max)
            else:
                kr_max = np.nanmax(self.k_r_arr)

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


        print('N_bins = %s' % N_bins)

        start0 = time.perf_counter()
        Power_spec1D = np.zeros(N_bins)
        kr_vec = np.zeros(N_bins)

        for i in range(len(k_r_bins)-1):

            # Calculating the radius:
            if log_bin_cond:
                kr_vec[i] = 10**(0.5*(np.log10(k_r_bins[i+1]) + np.log10(k_r_bins[i])))
            else:
                kr_vec[i] = ((k_r_bins[i+1] + k_r_bins[i])/2.0)

            # Defining the shell array index:
            shell_ind = np.logical_and(self.k_r_arr >= k_r_bins[i], self.k_r_arr <= k_r_bins[i+1])

            try:
                # Some bins don't contain data, this will ensure the script doesn't fail. These bins will be set
                # to NaN.
                if wedge_cond:
                    Power_spec1D[i] = np.average(self.power_cube[shell_ind],weights=self.weights_cube[shell_ind])
                else:
                    Power_spec1D[i] = np.average(self.power_cube[shell_ind],weights=self.weights_cube[shell_ind])
            except ZeroDivisionError:
                # For bins that don't have data we will set these to NaN.
                Power_spec1D[i] = np.nan
                
        end0 = time.perf_counter()
        
        print('1D PS calctime = %5.3f s' % (end0-start0))

        # Recalculating the field of view.
        #self.Omega_fov = (2*np.log(2))/(np.pi*(sig)**2) # Omega = 2 ln(2) / (pi* sig_grid^2) [Sr]
        self.Omega_fov = (2*np.log(2))/(np.pi*(sig)**2) # Omega = 2 ln(2) / (pi* sig_grid^2) [Sr]
        print('Sigma = %5.3f' % sig)
        print('Field of view %5.4f [Sr]' % self.Omega_fov)
        #self.Omega_fov = 0.0759 # Omega = 2 ln(2) / (pi* sig_grid^2) [Sr]


        # Cosmological unit conversion factor:
        dnu = self.dnu*1e+6 #[Hz] full bandwidth.
        dnu_f = self.dnu_f*1e+6 #[Hz] fine channel width.
        #Cosmo_factor = Power_spec.Power2Tb(dnu,dnu_f,self.nu_o,self.z,self.cosmo,self.Omega_fov)
        #Cosmo_factor = Power_spec.Power2Tb(dnu,dnu_f,self.nu_o,self.z,self.cosmo,self.Omega_fov)
        Cosmo_factor = (1/dnu_f**2)*Power_spec.Power2Tb(dnu,dnu_f,self.nu_o,self.z,self.cosmo,self.Omega_fov)

        print(f"Conversion Jy squared into P(k) is {Cosmo_factor:.5e}")

        self.Power1D = Power_spec1D*Cosmo_factor # [mK^3 Mpc^3 h^-3]
        self.k_r = kr_vec


    def Cylindrical(self):
        """
        Calculates the 2D cylindrical power spectra using the input Power object.
                
        Parameters
            ----------
            self : object
                Power object contains u and v arrays, as well as the observation redshift.
            kb : float
                Boltzman's constant.
            nu_21 : float
                21cm frequency in Hz.
            kb : float
                Speed of light km/s.
            
            Returns
            -------
        """

        bin_width = 2.5 # [lambda]
        
        # Converting into cosmological values.
        dk_r = Power_spec.uv_to_kxky(bin_width,self.z,self.cosmo) # h Mpc^-1
        k_perp_max = Power_spec.uv_to_kxky(self.uvmax,self.z,self.cosmo) # h Mpc^-1
        
        # Defininf the number of bins.
        N_bins = int(k_perp_max/dk_r)
        
        # Specifying the radius vector:
        kr_bins = np.linspace(0,k_perp_max,N_bins + 1)

        # Problems with instrument sampling. Ommiting first bin.
        kr_bins = kr_bins[1:]

        print(kr_bins)

        # The u_arr and v_arr should be shifted. 
        r_uv = np.sqrt(self.u_arr**2 + self.v_arr**2)
        kr_uv_arr = Power_spec.uv_to_kxky(r_uv,self.z,self.cosmo)

        # Initialising the power spectrum and radius arrays.
        Power_spec2D = np.zeros([len(self.eta),len(kr_bins)-1])
        kr_vec = np.zeros(len(kr_bins)-1)

        # Averaging the power in annular rings for each eta slice.
        for i in range(len(self.eta)):
            for j in range(len(kr_bins)-1):
                
                # Assigning the radius values. Needed for plotting purposes.
                kr_vec[j] = ((kr_bins[j+1] + kr_bins[j])/2.0)

                # Creating annunuls of boolean values.
                temp_ind = np.logical_and(kr_uv_arr >= kr_bins[j], kr_uv_arr <= kr_bins[j+1])

                # Weighted averaging of annuli values.
                Power_spec2D[i,j] = np.average(self.power_cube[temp_ind,i],weights=self.weights_cube[temp_ind,i]) ## Correct one.

        # Cosmological unit conversion factor:
        dnu = self.dnu*1e+6 #[Hz] full bandwidth.
        dnu_f = self.dnu_f*1e+6 #[Hz] fine channel width.
        #Cosmo_factor = Power_spec.Power2Tb(dnu,dnu_f,self.nu_o,self.z,self.cosmo,self.Omega_fov)
        #Cosmo_factor = Power_spec.Power2Tb(dnu,dnu_f,self.nu_o,self.z,self.cosmo,self.Omega_fov)
        #Cosmo_factor = Power_spec.Power2Tb(dnu,dnu_f,self.nu_o,self.z,self.cosmo,self.Omega_fov)
        Cosmo_factor = (1/dnu_f**2)*Power_spec.Power2Tb(dnu,dnu_f,self.nu_o,self.z,self.cosmo,self.Omega_fov)

        # Assigning the power.
        self.Power2D = Power_spec2D*Cosmo_factor # [mK^3 Mpc^3 h^-3]

        # Assigning the perpendicular and parallel components of the power spectrum.
        self.kperp = kr_vec
        self.kpar = Power_spec.eta_to_kz(self.eta,self.z,self.cosmo)

    @staticmethod
    def plot_spherical(k_r,Power1D,figsize=(8,6),xlim=None,ylim=None,title=None,figaxs=None,\
        xlabel=None,ylabel=None,step=True,**kwargs):
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

    @staticmethod
    def plot_cylindrical(Power2D,kperp,kpar,figsize=(7.5,10.5),cmap='viridis',
        name=None,xlim=None,ylim=None,vmin=None,vmax=None,clab=None,lognorm=True,
        title=None,horizon_cond=False,**kwargs):

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

        fov = 0.076
        
        # Performing a rudimentary fov calculation:
        # Not 100% sure these are correct, but they create cuts similar to Trott et al 2020.
        sig = 4 # lambda
        sig = 2 # lambda
        FWHM = 2*np.sqrt(2*np.log(2))/sig
        fov = FWHM**2 # rad**2


        #print(fov)

        fig, axs = plt.subplots(1, figsize = figsize, dpi=75, constrained_layout=True)

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



class Skew_spec:
    """
    This class defines the 1D and 2D skew spectrum. The 1D skew spectrum is a spherical average of the total
    (delta(k))^2 * delta(-k) field. The 2D skew spectrum is cylindrically averaged. This class also provides
    functions for plotting the 1D and 2D skew spectrum, as well as unit and coordinate transformations from
    Jy Hz to Tb, and (u,v,eta) to (kx,ky,kz). This class is largelly based off of the 'Power_spec' class.

    ...

    Attributes
    ----------
    fine : numpy array
        ...


    Methods
    -------
    Skew2Tb(freqs=""):
        ...
    uv_to_kxky(freqs=""):
        ...
    eta_to_kz(freqs=""):
        ...
    wedge_factor()
        ...
    Spherical()
        ...
    Cylindrical()
        ...
    plot_spherical()
        ...
    plot_cylindrical()
    """

    # Constants
    c = 299792458.0/1000 #[km/s]
    nu_21 = (1000*c)/(0.21) #[Hz]
    kb = 1380.649 # [Jy m^2 Hz K^-1] Boltzmann's constant.
    
    def __init__(self,S_cube,S_sqd_cube,eta,u_arr,v_arr,nu_o,dnu,dnu_f,
                weights_cube=None,nu_21=nu_21,cosmo=None):
        
        # Attributes will include data cube, eta, 
        # For determining the 2D bins.
        self.uvmax = 300

        self.Skew_cube = S_sqd_cube*np.conjugate(S_cube) # [Jy^3 Hz^2]

        test_cond = False
        if test_cond:
            # Pipeline testing purposes.
            out_path = '/home/jaiden/Documents/Skewspec/output/'
            out_name = 'Skew_test_cube'
            np.savez_compressed(out_path + out_name, Skew_cube = self.Skew_cube, Weights = weights_cube)
        else:
            pass

        if np.any(weights_cube):
            # Case for user inputted weigth cube.
            self.weights_cube = weights_cube

        else:
            print('Natural weighting case.')
            # Default weighting scheme is natural. Default set to not break older code.
            self.weights_cube = np.zeros(np.shape(S_cube))
            
            # Only cells with values are assigned weights.
            self.weights_cube[self.Skew_cube > 0.0] = 1.0

            #print(np.sum(self.weights_cube))
        
        self.eta = eta # [Hz^-1]
        self.u_arr = u_arr # Should have units of wavelengths.
        self.v_arr = v_arr # Should have units of wavelengths.
        self.nu_o = nu_o # [Hz]
        self.z = (nu_21/self.nu_o) - 1
        self.dnu = dnu # Bandwidth in [MHz].
        self.dnu_f = dnu_f # Fine channel width in [MHz].
        print('Redshift = %5.2f' % self.z)

        if cosmo != None:
            # User inputted cosmology.
            print('Using non-default cosmology.')
            self.cosmo = cosmo
        else:
            # Default is the Plank18 cosmology.
            from astropy.cosmology import Planck18

            self.cosmo = Planck18

        # Save memory.
        del S_cube
        del S_sqd_cube
    
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
        from scipy import signal

        # Constants
        c = 299792458.0/1000 #[km/s]
        nu_21 = c*1000/(0.21) #[Hz]
        kb = 1380.649 # [Jy m^2 K^-1] Boltzmann's constant.

        # Constants.
        lam_o = 1000*c/nu_o #[m]
        
        # Future versions we will have the field of view as an input. 
        #fov = 0.076 # [sr] field of view. Approximate.
        #fov = 0.095 # [sr] field of view. Approximate.
        N_chans = dnu/dnu_f

        if verbose:
            print('Observed wavelength = %5.3f [m]' % lam_o)
            print('Fine channel width = %5.3e' % dnu_f)
        else:
            pass
        
        # Calculating the volume correction factor:
        window = signal.blackmanharris(int(dnu/dnu_f))
        Ceff = np.sum(window)/(dnu/dnu_f)

        if verbose:
            print('Volume correction factor = %5.3f' % (Ceff))
        else:
            pass

        # Cosmological scaling parameter:
        h = cosmo.H(0).value/100
        #print('Harry %5.3f'%h)
        E_z = cosmo.efunc(z)

        # Cosmological distances:
        Dm = cosmo.comoving_distance(z).value*h #[Mpc/h]
        #print('Co-moving distance = %5.3f [Mpc h^-1]' % Dm)
        DH = 3000 # [Mpc/h] Hubble distance.

        # Volume term.
        co_vol = (1/Ceff)*(Dm**2 * DH *(1 + z)**2)/(nu_21 * E_z) # [sr^-1 Hz^-1 Mpc^3 h^-3]

        if verbose:
            print('Squared co-moving volume term = %5.3f [sr^-1 Hz^-1 Mpc^3 h^-3]' % co_vol)
            #print('Squared co-moving volume term = %5.3f [sr^-2 Hz^-2 Mpc^6 h^-6]' % co_vol)
        else:
            pass

        # Converting a 1 Jy^2 source to K^3 Mpc^6 h^-6.
        conv_factor =  dnu * (lam_o**6/(8*kb**3)) * (1/(Omega_fov)) * co_vol * 1e+9 # [mK^3 Mpc^3 h^-3]
        #conv_factor =  dnu * (lam_o**6/(8*kb**3)) * (1/(Omega_fov**2)) * co_vol * 1e+9 # [mK^3 Mpc^3 h^-3]

        if verbose:
            print('Conversion factor = %5.3e [mK^3 Hz^-3 Mpc^3 h^-3]' % conv_factor)
            print('Conversion factor = %5.3e [mK^3 Hz^-3 Mpc^3]' % (conv_factor*h**3))
        else:
            pass
        
        return conv_factor

    @staticmethod
    def uv_to_kxky(u,z,cosmo):
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
            
            Returns
            -------
            k_vec : numpy array, float
                NDarray of k-mode values. Should be in units of h*Mpc^-1. 
        """

        # Cosmological scaling parameter:
        h = cosmo.H(0).value/100 # Hubble parameter.
    
        # Cosmological distances:
        Dm = cosmo.comoving_distance(z).value*h #[Mpc/h] Transverse co-moving distance.

        # Converting u to k
        k_vec = u * (2*np.pi/Dm) # [Mpc^-1 h]

        return k_vec

    @staticmethod
    def eta_to_kz(eta,z,cosmo):
        """
        Convert eta into k_z as per Morales et al. (2004).
        Uses the Plank 2018 cosmology as default.
                
        Parameters
            ----------
            eta : numpy array, float
                1Darray of eta values. 
            z : float
                Redshift at the central frequency of the band.
            
            Returns
            -------
            k_z : numpy array, float
                1Darray of kz values. Should be in units of h*Mpc^-1.
        """

        # Constant:
        c = 299792458.0/1000 #[km/s]
        nu_21 = (1000*c)/(0.21) #[Hz]

        # Cosmological scaling parameter:
        E_z = cosmo.efunc(z) ## Scaling function, see (Hogg 2000)

        # Cosmological distances:
        DH = 3000 # c/Ho [Mpc/h] Hubble distance.

        # k_||
        k_z = eta * (2*np.pi*nu_21*E_z)/(DH*(1 + z)**2) # [Mpc^-1 h]

        return k_z

    @staticmethod
    def wedge_factor(z,cosmo):
        """
        Nicholes horizon cosmology cut.
                
        Parameters
            ----------
            z : float
                Redshift.
            cosmo : Astropy Object
                Astropy cosmology object, default is None. If None use Plank18 cosmology.
            
            Returns
            -------
            wedge_factor : float
                k|| > wedge_factor * k_perp cut.
        """

        # Cosmological scaling parameter:
        h = cosmo.H(0).value/100 # Hubble parameter.
        E_z = cosmo.efunc(z) ## Scaling function, see (Hogg 2000)

        # Cosmological distances:
        Dm = cosmo.comoving_distance(z).value*h #[Mpc/h] Transverse co-moving distance.
        DH = 3000 # [Mpc/h] Hubble distance.

        wedge_factor = Dm*E_z/(DH*(1 + z)) 

        return wedge_factor

    def Spherical(self,wedge_cond=False,N_bins=60,log_bin_cond=False,kr_min=None,kr_max=None,
                horizon_cond=True,wedge_cut=None,sig=2.15):
        """
        Calculates the 1D spherical power spectra using the input Power object.
                
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

        # Defining the kx, ky, and kz values from u,v and eta.
        k_z = Skew_spec.eta_to_kz(self.eta,self.z,self.cosmo) # [Mpc^-1 h]
        k_x = Skew_spec.uv_to_kxky(self.u_arr,self.z,self.cosmo) # [Mpc^-1 h]
        k_y = Skew_spec.uv_to_kxky(self.v_arr,self.z,self.cosmo) # [Mpc^-1 h]

        # Creating 3D k_r array.
        self.k_r_arr = np.array([np.sqrt(k_x**2 + k_y**2 + kz**2) for kz in k_z]).T

        if wedge_cond:
            # Condition for ignoring the wedge contribution to the power spectrum.
            
            # If the grid kernel sixe (sig) provided calculate the FoV.
            self.Omega_fov = (2*np.log(2))/(np.pi*(sig)**2) # Omega = 2 ln(2) / (pi* sig_grid^2) [Sr]

            # Creating the k_perp array.
            k_perp = np.sqrt(k_x**2 + k_y**2) # [Mpc^-1 h]
            # Specifying a minimum k_perp.
            k_perp_min = 0.1 # [Mpc^-1 h]
            
            if horizon_cond:
                grad = 0.5*np.pi # Horizon cut gradient.
            else:
                grad = 0.5*np.sqrt(self.Omega_fov) # Beam FoV cut.

            if wedge_cut:
                # Option to manually input a wedge cut value.
                wedge_cut = wedge_cut
            else:
                # Default is to calculate the horizon or beam grad.
                wedge_cut = grad*Skew_spec.wedge_factor(self.z,self.cosmo) # Nicholes horizon cosmology cut.
            
            print('wedge_cut %5.3f' % wedge_cut)

            # Calculating the wedge mask array.
            wedge_ind_cube = np.array([np.logical_or(k_par < wedge_cut*k_perp, k_perp >= k_perp_min) for k_par in k_z]).T
            #wedge_ind_cube = np.array([k_par < wedge_cut*k_perp for k_par in k_z]).T

            if kr_min:
                # User can manually input a kr min.
                kr_min = float(kr_min)
            else:
                ## Testing
                kr_min = 0.1

            # Setting all k_par modes greater than some mode set to True.
            wedge_ind_cube[:,:,k_z < kr_min] = True
            
            ## 
            # Setting the wedge to zero.
            self.Skew_cube[wedge_ind_cube] = np.NaN
            self.weights_cube[wedge_ind_cube] = np.NaN
            self.k_r_arr[wedge_ind_cube] = np.NaN

            # Calculating kr_max.
            kr_max = np.nanmax(self.k_r_arr)

        else:

            if kr_min:
                # User can manually input a kr min.
                kr_min = float(kr_min)
            else:
                kr_min = np.nanmin(self.k_r_arr[self.k_r_arr > 0.0])

            if kr_max:
                # User can manually input a kr max.
                kr_max = float(kr_max)
            else:
                kr_max = np.nanmax(self.k_r_arr)

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


        print('N_bins = %s' % N_bins)

        start0 = time.perf_counter()
        #Skew_spec1D = np.zeros(N_bins)
        # Should be complex.
        Skew_spec1D = np.zeros(N_bins,dtype=complex)
        kr_vec = np.zeros(N_bins)

        for i in range(len(k_r_bins)-1):

            # Calculating the radius:
            if log_bin_cond:
                kr_vec[i] = 10**(0.5*(np.log10(k_r_bins[i+1]) + np.log10(k_r_bins[i])))
            else:
                kr_vec[i] = ((k_r_bins[i+1] + k_r_bins[i])/2.0)

            # Defining the shell array index:
            shell_ind = np.logical_and(self.k_r_arr >= k_r_bins[i], self.k_r_arr <= k_r_bins[i+1])

            try:
                # Some bins don't contain data, this will ensure the script doesn't fail. These bins will be set
                # to NaN.
                if wedge_cond:
                    Skew_spec1D[i] = np.average(self.Skew_cube[shell_ind],weights=self.weights_cube[shell_ind])
                else:
                    Skew_spec1D[i] = np.average(self.Skew_cube[shell_ind],weights=self.weights_cube[shell_ind])
            except ZeroDivisionError:
                # For bins that don't have data we will set these to NaN.
                Skew_spec1D[i] = np.nan
                
        end0 = time.perf_counter()
        
        print('1D PS calctime = %5.3f s' % (end0-start0))

        self.Omega_fov = (2*np.log(2))/(np.pi*(sig)**2) # Omega = 2 ln(2) / (pi* sig_grid^2) [Sr]

        print('Field of view %5.4f [Sr]' % self.Omega_fov)

        # Cosmological unit conversion factor:
        dnu = self.dnu*1e+6 #[Hz] full bandwidth.
        dnu_f = self.dnu_f*1e+6 #[Hz] fine channel width.

        Skew_spec1D = Skew_spec1D/(dnu_f**2) 
        
        Cosmo_factor = Skew_spec.Skew2Tb(dnu,dnu_f,self.nu_o,self.z,self.cosmo,self.Omega_fov,verbose=True)
        print('Cosmo-conversion-factor = %5.3e' % Cosmo_factor)
        print('Mean skew spec = %5.3e [Jy^3 Hz^2]' % np.nanmean(Skew_spec1D))
        print('Max skew spec = %5.3e [Jy^3 Hz^2]' % np.nanmax(Skew_spec1D))

        self.Skew1D = Skew_spec1D*Cosmo_factor # [mK^3 Mpc^3 h^-3]
        self.k_r = kr_vec


    def Cylindrical(self):
        """
        Calculates the 2D cylindrical power spectra using the input Power object.
                
        Parameters
            ----------
            self : object
                Power object contains u and v arrays, as well as the observation redshift.
            kb : float
                Boltzman's constant.
            nu_21 : float
                21cm frequency in Hz.
            kb : float
                Speed of light km/s.
            
            Returns
            -------
        """

        bin_width = 2.5 # [lambda]
        
        # Converting into cosmological values.
        dk_r = Skew_spec.uv_to_kxky(bin_width,self.z,self.cosmo) # h Mpc^-1
        k_perp_max = Skew_spec.uv_to_kxky(self.uvmax,self.z,self.cosmo) # h Mpc^-1
        
        # Defininf the number of bins.
        N_bins = int(k_perp_max/dk_r)
        
        # Specifying the radius vector:
        kr_bins = np.linspace(0,k_perp_max,N_bins + 1)

        # Problems with instrument sampling. Ommiting first bin.
        kr_bins = kr_bins[1:]

        # The u_arr and v_arr should be shifted. 
        r_uv = np.sqrt(self.u_arr**2 + self.v_arr**2)
        kr_uv_arr = Skew_spec.uv_to_kxky(r_uv,self.z,self.cosmo)

        # Initialising the power spectrum and radius arrays.
        Skew_spec2D = np.zeros([len(self.eta),len(kr_bins)-1])
        kr_vec = np.zeros(len(kr_bins)-1)

        # Averaging the power in annular rings for each eta slice.
        for i in range(len(self.eta)):
            for j in range(len(kr_bins)-1):
                
                # Assigning the radius values. Needed for plotting purposes.
                kr_vec[j] = ((kr_bins[j+1] + kr_bins[j])/2.0)

                # Creating annunuls of boolean values.
                temp_ind = np.logical_and(kr_uv_arr >= kr_bins[j], kr_uv_arr <= kr_bins[j+1])

                # Weighted averaging of annuli values.
                Skew_spec2D[i,j] = np.average(self.Skew_cube[temp_ind,i],weights=self.weights_cube[temp_ind,i]) ## Correct one.

        # Cosmological unit conversion factor:
        dnu = self.dnu*1e+6 #[Hz] full bandwidth.
        dnu_f = self.dnu_f*1e+6 #[Hz] fine channel width.
        Cosmo_factor = Skew_spec.Skew2Tb(dnu,dnu_f,self.nu_o,self.z,self.cosmo,self.Omega_fov)

        Skew_spec2D = Skew_spec2D/(dnu_f**2)
        # Assigning the power.
        self.Skew2D = Skew_spec2D*Cosmo_factor # [mK^3 Mpc^3 h^-3]

        # Assigning the perpendicular and parallel components of the power spectrum.
        self.kperp = kr_vec
        self.kpar = Skew_spec.eta_to_kz(self.eta,self.z,self.cosmo)


    @staticmethod
    def plot_spherical(k_r,Skew_spec1D,figsize=(8,6),xlim=None,ylim=None,title=None,figaxs=None,\
        xlabel=None,ylabel=None,step=True,symlog_cond=True,sym_log_scale=10,linscaley=1,**kwargs):
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

        # Change this to be optional depending on whether there are negative values or not.
        plt.loglog()
        if symlog_cond:
            plt.yscale('symlog', linthresh=sym_log_scale*np.min(np.abs(Skew_spec1D)),linscaley=linscaley)
        else:
            pass

        if step:
            # Default options is a step plot.
            axs.step(k_r,Skew_spec1D,**kwargs)
        else:
            # Line plot is more useful for comparing Power spectra with different bin sizes.
            axs.plot(k_r,Skew_spec1D,**kwargs)


        if xlim:
            axs.set_xlim(xlim)
        if ylim:
            axs.set_ylim(ylim)
            if symlog_cond:
                axs.set_yscale('symlog')
            else:
                pass

        if xlabel:
            axs.set_xlabel(xlabel,fontsize=24)
        else:
            axs.set_xlabel(r'$|\mathbf{k}| \,[\it{h}\rm{\,Mpc^{-1}}]$',fontsize=24)

        if ylabel:
            axs.set_ylabel(ylabel,fontsize=24)
        else:
            axs.set_ylabel(r'$S(k) \, [\rm{mK^3}\,\it{h^{-3}}\,\rm{Mpc^3}]$',fontsize=24)

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


    @staticmethod
    def plot_cylindrical(Skew_spec2D,kperp,kpar,figsize=(7.5,10.5),cmap='viridis',
        name=None,xlim=None,ylim=None,vmin=None,vmax=None,clab=None,lognorm=True,
        title=None,horizon_cond=False,linthresh=None,linscale=None,**kwargs):
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
        import matplotlib

        fov = 0.076
        
        # Performing a rudimentary fov calculation:
        # Not 100% sure these are correct, but they create cuts similar to Trott et al 2020.
        sig = 4 # lambda
        sig = 2 # lambda
        FWHM = 2*np.sqrt(2*np.log(2))/sig
        fov = FWHM**2 # rad**2


        #print(fov)

        fig, axs = plt.subplots(1, figsize = figsize, dpi=75, constrained_layout=True)

        if vmax:
            vmax=vmax
        else:
            pspec_max = np.max(np.log10(Skew_spec2D[Skew_spec2D > 0]))
            vmax = 10**pspec_max
        
        if vmin != None:
            # If vmin=0 this is considered false. So condition has to be different
            # to vmax.
            vmin=vmin
        else:
            pspec_min = np.min(np.log10(Skew_spec2D[Skew_spec2D > 0]))

            vmin = 10**pspec_min

        if lognorm:
            #norm = matplotlib.colors.LogNorm()
            norm = matplotlib.colors.SymLogNorm(linthresh=linthresh,linscale=linscale,base=10)
        else:
            norm = None

        print('Min = %5.3e' % np.min(Skew_spec2D[Skew_spec2D > 0]))
        print('Max = %5.3e' % np.max(Skew_spec2D[Skew_spec2D > 0].flatten()[0]))
        
        # Setting NaN values to a particular colour:
        #cmap = matplotlib.cm.viridis
        cmap = matplotlib.cm.get_cmap("Spectral_r")
        cmap.set_bad('lightgray',1.)
    

        im = axs.imshow(Skew_spec2D,cmap=cmap,origin='lower',\
                extent=[np.min(kperp),np.max(kperp),np.min(kpar),np.max(kpar)],\
                norm=norm,vmin=vmin,vmax=vmax, aspect='auto',**kwargs)
        

        # Setting the colour bars:
        cb = fig.colorbar(im, ax=axs, fraction=0.04, pad=0.002, extend='both')

        if clab:
            cb.set_label(label=clab,fontsize=20)
        else:
            cb.set_label(label=r'$S(k_\perp,k_{||}) \, [\rm{mK^3}\,\it{h^{-3}}\,\rm{Mpc^3}]$',fontsize=20)
        
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