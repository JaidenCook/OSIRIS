# OSIRIS
Observational Supernova-remnant Instrumental Reionisation Investigative Simulator


## Goal of This Work

This work aims to simulate the visibilities measured for interferometers, specifically the Murchison widefield array (MWA). This work is particularly focused on modelling the visibilities of supernova remnants (SNRs) and other extended radio sources, to investigate their effects on detecting the 21cm epoch of reionisation (EoR) signal. This is done by simulating a true sky brightness distribution $I(l,m)$, which is then fast fourier transformed (FFT) into the true sky visibilities $\mathcal{V}(u,v)$. Using a Gaussian kernel we then degrid the trus sky brightness visibilities to simulate what the MWA would measure for a particular observation. These are then regridded and FFT in the frequency domain to obtain the 2D power spectrum.


### Note:

Pipeline usage is currently lacking documentation, if anyone actually wants to use this pipeline email me at Jaiden.cook@postgrad.curtin.edu.au.
