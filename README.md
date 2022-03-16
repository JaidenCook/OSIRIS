# OSIRIS
Observational Supernova-remnant Instrumental Reionisation Investigative Simulator


## Goal of This Work

This work aims to simulate the visibilities measured for interferometers, specifically the Murchison widefield array (MWA). This work is particularly focused on modelling the visibilities of supernova remnants (SNRs) and other extended radio sources, to investigate their effects on detecting the 21cm epoch of reionisation (EoR) signal. This is done by simulating a true sky brightness distribution $I(l,m)$, which is then fast fourier transformed (FFT) into the true sky visibilities $\mathcal{V}(u,v)$. Using a Gaussian kernel we then degrid the trus sky brightness visibilities to simulate what the MWA would measure for a particular observation. These are then regridded and FFT in the frequency domain to obtain the 2D power spectrum.


### Note:

Pipeline usage is currently lacking documentation, if anyone actually wants to use this pipeline email me at Jaiden.cook@postgrad.curtin.edu.au.

![Pipeline-flow-chart](https://user-images.githubusercontent.com/43106834/158518789-ac0d5416-4f02-4ca0-a929-7b511b59f8a7.png)


## Supernova Remnant and Centaurus A Model

The FITS file containing the Gaussian component models for the SNRs and Centaurus A can be found in the ```./pipeline``` directory, where the file is named ```CenA-GP-gauss_model.fits```. The FITS file is a FITS table format, with the table columns being ```Name,RA,u_RA,DEC,u_DEC,Sint,u_Sint,Maj,u_Maj,Min,u_Min,PA,u_PA,alpha,ModelID```. For the ```Name``` columns is the same as the one from Greens catalogue which you can find here ```https://www.mrao.cam.ac.uk/surveys/snrs/```, with the exception of the CenA model components which are simply named 'cenA'. Each model has an associated ID which is stored in the ```ModelID``` column. This ID should match the ID given in greens catalogue, with the inclusion of 'cenA' which has the ID 295. Each row of the table represents a different Gaussian, where each Gaussian model component can be grouped by name or by their model ID. Therefore the Gaussian parameters for their centre, their size, their position angle and their amplitude are given by ```RA,DEC,Sint,Maj,Min,PA``` with their associated error columns designated by the prefix ```u_```.

### Caveats Regarding the Model
All flux densities are the expected 200 MHz flux density. For SNRs that were not fit, we simply used the existing Major, Minor SI and Sint from Green's catalogue. Green's catalogue does not provide a position angle for the elliptical sizes fo the Gaussians, so use these with Caution. These sources have PA=0. 

The position angle for each component will likely need to be rotated, the Gaussian models were fit in pixel coordinates. Using the software ```Topcat``` I found that in order to get the correct rotation I had to apply this formula ```PA = 360 - (PA +90) = 270-PA```, this provides the correct rotation with respect to the celestial coordinate frame. 

Some of the errors for the CenA model components are incorrect. This is true of the components which have the model ID 298, and 299. This corresponds to the outer lobes of Centaurus A. When fitting these lobes I rescaled and downsampled the images, I forgot to rescale the errors for the integrated flux density. To properly scale them they need to be divided by (19x19).

