# OSIRIS
Observational Supernova-remnant Instrumental Reionisation Investigative Simulator


## Goal of This Work

This work aims to simulate the visibilities measured for interferometers, specifically the Murchison widefield array (MWA). This work is particularly focused on modelling the visibilities of supernova remnants (SNRs) and other extended radio sources, to investigate their effects on detecting the 21cm epoch of reionisation (EoR) signal. This is done by simulating a true sky brightness distribution _I_(l,m), which is then fast fourier transformed (FFT) into the true sky visibilities _V_(u,v). Using a Gaussian kernel we then degrid the trus sky brightness visibilities to simulate what the MWA would measure for a particular observation. These are then regridded and FFT in the frequency domain to obtain the 2D power spectrum.


## Pipeline

![Pipeline-flow-chart](https://user-images.githubusercontent.com/43106834/158518789-ac0d5416-4f02-4ca0-a929-7b511b59f8a7.png)

The pipeline scripts can be found in the directory ```./pipeline```. The main script ```SNR_pipeline.py``` is capable of running the entire pipeline, however, there is an option to provide the pipeline with a sky-model image cube with the format ```.npz```, this is a compressed ```numpy``` format. This is the preferred operating mode. The sky-model image cubes can be generated from ```SNR_sky-model.py```. This script takes an input OBSID which is the GPS time for a particular MWA observation. The GPS time is used to determine which sources in the sky-model are above the horizon. This script then constructs the sky-model image cube using the FITS file ```CenA-GP-gauss_model-V2.fits``` which contains all the SNR and Centaurus A models. This script also has options for excluding known bright sources such as Centaurus A, Pupis A, Vela and the Crab nebula. Future versions of this script will allow users to input a particular model ID if the user wishes to exclude a particular source.

Once the model is generated it can then be passed to ```SNR_pipeline.py``` which will then FFT, and average spherically and cylindrically to determine the 1D and 2D power spectrum. The power spectrum is output as a ```.npz``` file. The 1D and 2D Power spectrum can then be read and plotted using the script ```SNR_Ratio_plot.py```. See ```--help``` on these scripts for more information about the available options.

## Command Line Script Examples

The pipeline scripts require the metafits files for a real observation. These can be downloaded from the MWA ASVO website ```https://asvo.mwatelescope.org/```. The metafits file is used to get the delays and the frequency channels for the observation. Alternatively in future versions these can be input parameters, or a single metafits file can be modified to have the desired MWA delays and frequency channels. 

### Example SNR_sky-model.py
```python SNR_sky-model.py --obsid 1080136736 --no_cenA --all_sky --save_plots --kernel_size 91```

Outputs file ```1080136736_all-sky_no-cenA.npz```. Adding the options ```--save_partial_models``` and ```--S_app``` will generate the approximate 10%, 50%, 90% and 100% apparent sky-model cubes.

### Example SNR_pipeline.py
```python SNR_pipeline.py --obsid 1080136736 --sky_model "../models/1080136736_all-sky_no-cenA.npz" --obs_length=10000 --beam_interp --no_wedge --kernel_size=91 --grid_kernel_size=91 --sigma_grid=4```

Outputs file ```1080136736_all-sky_no-cenA_tobs10000_no-wedge.npz```

### Example SNR_Ratio_plot.py

```python SNR_Ratio_plot.py --2Dpspec1 "1080136736_all-sky_no-cenA_tobs10000_no-wedge.npz" --outname "1080136736_all-sky_no-cenA-2Dpspec" --max=1e+12 --min=1e+5```

outputs file ```1080136736_all-sky_no-cenA-2Dpspec.png```

For questions on how to setup and run the pipeline please contact me at ```Jaiden.cook@postgrad.curtin.edu.au```.

## Supernova Remnant and Centaurus A Model

The FITS file containing the Gaussian component models for the SNRs and Centaurus A can be found in the ```./pipeline``` directory, where the file is named ```CenA-GP-gauss_model-V2.fits```. The FITS file is a FITS table format, with the table columns being ```[Name,RA,u_RA,DEC,u_DEC,Sint,u_Sint,Maj,u_Maj,Min,u_Min,PA,u_PA,alpha,ModelID]```. For the ```Name``` columns is the same as the one from Greens catalogue which you can find here ```https://www.mrao.cam.ac.uk/surveys/snrs/```, with the exception of the CenA model components which are simply named 'cenA'. Each model has an associated ID which is stored in the ```ModelID``` column. This ID should match the ID given in greens catalogue, with the inclusion of 'cenA' which has the ID 295. Each row of the table represents a different Gaussian, where each Gaussian model component can be grouped by name or by their model ID. Therefore the Gaussian parameters for their centre, their size, their position angle and their amplitude are given by ```RA,DEC,Sint,Maj,Min,PA``` with their associated error columns designated by the prefix ```u_```.

### Caveats Regarding the Model
All flux densities are the expected 200 MHz flux density. For SNRs that were not fit (because they were too faint, small or high in declination), we simply used the existing Major, Minor, SI (spectral index) and Sint from Green's catalogue. Green's catalogue does not provide a position angle for the elliptical sizes for the Gaussians, so use these with Caution. These sources have PA=0. 

The position angle for each component will likely need to be rotated, the Gaussian models were fit in pixel coordinates. Using the software ```Topcat``` I found that in order to get the correct rotation relative to the celestial frame I had to apply this formula ```PA = 360 - (PA + 90) = 270 - PA```. 
