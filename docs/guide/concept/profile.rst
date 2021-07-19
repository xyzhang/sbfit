********
Profiles
********


Different from X-ray spectra, which are usually
presented without deconvolution. In imaging analysis, people always show
'surface brightness' profiles rather than 'count' profiles. 

Channel and bin
===============

Channel is the smallest unit of a profile. The width of the channel should be smaller
than the PSF width. The surface brightness in the $i$th channel is

.. math::

   S_i = \frac{N_{\mathrm{total},i} - N_{\mathrm{NXB},i}}
   {A_\mathrm{pixel}\sum_{j}{t_{i,j}} }

To calculate the surface brightness in each channel, the package first
calculate the total count number, scaled NXB count number, and the sum of 
exposure in each channel. When fitting a profile, the model is calculated in
each channel as well.

In most cases, the count number in each channel is small. In other words, 
the raw profile is oversampled. Although the fit results will be the same when 
use the C-statistics, to better present the profile, rebinning is needed.



+-------------------+-------------------------------------------+
|Column             |Description                                |
+===================+===========================================+
|``r``              |Bin radius                                 |
+-------------------+-------------------------------------------+
|``r_error_left``   |Left bin width                             |
+-------------------+-------------------------------------------+
|``r_error_right``  |Right bin width                            |
+-------------------+-------------------------------------------+
|``sb``             |Observed surface brightness                |
+-------------------+-------------------------------------------+
|``sb_error``       |Observed surface brightness uncertainty    |
+-------------------+-------------------------------------------+
|``bkg_sb``         |Subtracted NXB profile                     |
+-------------------+-------------------------------------------+
|``bkg_sb_error``   |NXB profile uncertainty                    |
+-------------------+-------------------------------------------+
|``model_sb``       |Model profile                              |
+-------------------+-------------------------------------------+
|``total_cts``      |Total count number                         |
+-------------------+-------------------------------------------+
|``bkg_cts``        |Scaled NXB count number                    |
+-------------------+-------------------------------------------+


Profile types
=============

There are two types of surface brightness profiles:

- Radial profile
- Azimuthal profile


Binning methods
===============

The ``sbfit`` package supports three binning methods:

- Minimum total count
- Linear step
- Logarithmic step