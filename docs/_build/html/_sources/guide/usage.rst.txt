*****
Usage
*****

.. note:: This page only provides basic usages of the package.

Load the package.

.. code-block:: python

   import sbfit


Load data and extract profiles
==============================

Load images
-----------
A set of count map, exposure map, and NXB map will be loaded to an :py:class:`~sbfit.observation.Observation` object.

.. code-block:: python

   obs1 = sbfit.Observation("countmap1.img", "expmap1.img", "nxbmap1.img")
   obs2 = sbfit.Observation("countmap2.img", "expmap2.img", "nxbmap2.img")

Then we load all image sets into an
:py:class:`~sbfit.observation.ObservationList` object, where we subtract surface brightness profiles.

.. code-block:: python

   obs_list = sbfit.ObservationList([obs1, obs2])

Alternatively, we can start with an empty :py:class:`~sbfit.observation.ObservationList` object and add images into it.

.. code-block:: python

   obs_list = sbfit.ObservationList()
   obs_list.add_observation_from_file("countmap1.img", "expmap1.img", "nxbmap1.img")
   obs_list.add_observation_from_file("countmap2.img", "expmap2.img", "nxbmap2.img")

Load regions
------------
We use :py:func:`~sbfit.region.read_region` to read a DS9 format region file and
return a :py:class:`~sbfit.region.RegionList` object, which will be used later
for extracting a surface brightness profile.

.. code-block:: python

   reg_list = sbfit.read_region("ds9.reg")

Extract profiles
----------------
We use a :py:class:`~sbfit.region.RegionList` object to extract a
:py:class:`~sbfit.profile.Profile` object from an
:py:class:`~sbfit.observation.ObservationList` object using the
:py:meth:`~sbfit.observation.ObservationList.get_profile` method.

.. code-block:: python

   pro = obs_list.get_profile(reg_list)


Model fitting and visualization
===============================

Rebin
-----

The first step of analyzing a profile is to select a radius range and rebin it
using the :py:meth:`~sbfit.profile.Profile.rebin` method.
For example, we select a 10 arcsec to 100 arcsec range and rebin the profile
to let each bin have at least 100 total counts. Channels outside
the selected range are ignored.

.. code-block:: python

   pro.rebin(10, 100, method="min_cts", min_cts=100)

Display a profile
-----------------

The profile can be easily plotted using the
:py:meth:`~sbfit.profile.Profile.plot` method.

.. code-block:: python

   pro.plot(type="profile")

Model fitting
-------------

A model can be set using the :py:meth:`~sbfit.profile.Profile.set_model` method.
For example, we set a :py:class:`~sbfit.model.Beta` model to the profile object
``pro``.

.. code-block:: python

   beta = sbfit.model.Beta()
   pro.set_model(beta)

To ensure that the optimizer works well, we need to adjust the initial parameters to make the model
profile close to the observed profile. The method
:py:meth:`~sbfit.profile.Profile.calculate` is used to calculate the model
profile using current parameters. Users can use the
:py:meth:`~sbfit.profile.Profile.calculate` and :py:meth:`~sbfit.profile.Profile.plot`
methods repeatly to finish this step.

.. code-block:: python

   # set parameters
   pro.model.norm = 1e-2
   pro.model.beta = 0.6
   pro.model.r = 50

   # calculate model parameter
   pro.calculate()

   # plot the model profile together with the observed profile
   pro.plot()

The last step before fitting is to set boundaries for the free parameters.

.. code-block:: python

   pro.model.norm.bounds = (5e-3, 2e-2)
   pro.model.beta.bounds = (0.4, 0.9)
   pro.model.r.bounds = (30, 80)

.. warning:: This is a necessary step to prevent the optimizer from
   being crazy.

Finally, we can use the method :py:meth:`~sbfit.profile.Profile.fit` to fit
the model to the observed profile. An implemented Levenberg-Marquardt optimizer
will minimize the c-stat value until the decrement is less than a given
tolerance.

.. code-block:: python

   pro.fit(show_step=True, tolerance=0.01)

The L-M optimizer also estimates first-order uncertainties of the parameters,
which are stored as the attribute :py:attr:`~sbfit.profile.Profile.error_approx`.

Uncertainty estimation using MCMC
---------------------------------

To accurately estimate the uncertainties, we provide a method
:py:meth:`~sbfit.profile.Profile.mcmc_error` to call the
`emcee <https://github.com/dfm/emcee>`_ package. For example, we run a 10000 times
MCMC process and discard the first 1000 steps.

.. code-block:: python

   pro.mcmc_error(nsteps=10000, burnin=1000)

The MCMC chains and parameter contours can also be plotted using the
:py:meth:`~sbfit.profile.Profile.plot` method.

.. code-block:: python

   # plot MCMC chains
   pro.plot(type="mcmc_chain")

   # plot 'corner' contours
   pro.plot(type="contour")

The estimated uncertainties using MCMC method are stored in the attribute
:py:attr:`~sbfit.profile.Profile.error`.

Output profiles
===============

The profile data is stored as the attribute :py:attr:`~sbfit.profile.Profile.binned_profile`,
which is an ``astropy.table.Table`` object. The table data can be write to a
text file using the ``writeto()`` method.

.. code-block:: python

   # write to a text file
   pro.binned_profile.writeto("profile.txt", format="ascii.fixed_width",
                              overwrite=True)

   # write to a fits file
   pro.binned_profile.writeto("profile.fits", overwrite=True)

