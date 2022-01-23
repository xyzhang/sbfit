"""
This module includes observation classes.
"""

import numpy as np
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy import units as u

from .image import CtsImage, ExpImage, BkgImage
from .profile import Profile
from .surface import Surface
from .region import RegionList, ExcludeRegion
from .utils import get_pixel_scale

__all__ = ["Observation", "ObservationList"]


class Observation(object):
    """
    The class of a single observation, which contains a count map, an exposure
    map, and a background map.

    Parameters
    ----------
    cts_image_file : str
        The count image of the observation.
    exp_image_file : str
        The exposure map of the observation.
    bkg_image_file : str
        The image that represents the background level of the observation.
    bkg_norm_type : {"count", "flux"}, optional
        The type of the background scaling factor.
        "count" : the ratio between the total background count number in the
        count image and that in the background image.
        "flux" : the ratio between the total background count rate in the count
        image and that in the background image.
        For the "flux" type, the exposure time of the background image need to
        be properly provided. The default norm_type = "count".
    bkg_norm_keyword : str, optional
        The keyword of scaling factor in the background image header.
        The default norm_keyword = "bkgnorm".
    extension : int, optional
        The number of the hdu that contains the image data. The default
        extension = 0.
    """

    def __init__(self, cts_image_file, exp_image_file, bkg_image_file,
                 bkg_norm_type="count", bkg_norm_keyword="bkgnorm",
                 extension=0):
        self._cts_image = CtsImage(cts_image_file, extension=extension)
        self._exp_image = ExpImage(exp_image_file, extension=extension)
        self._bkg_image = BkgImage(bkg_image_file, norm_type=bkg_norm_type,
                                   norm_keyword=bkg_norm_keyword,
                                   extension=extension)
        if self._cts_image.shape == self._exp_image.shape \
                and self._cts_image.shape == self._bkg_image.shape:
            pass
        else:
            raise ValueError("All images must have the same size.")

    @property
    def cts_image(self):
        """The count image."""
        return self._cts_image

    @property
    def exp_image(self):
        """The exposure map."""
        return self._exp_image

    @property
    def bkg_image(self):
        """The background image."""
        return self._bkg_image

    @property
    def telescope(self):
        """The telescope by which the observation was performed."""
        try:
            telescope = self._cts_image.header["TELESCOP"]
        except KeyError:
            telescope = None
        return telescope

    @property
    def instrument(self):
        """The instrument by which the observation was performed."""
        try:
            instrument = self._cts_image.header["INSTRUME"]
        except KeyError:
            instrument = None
        return instrument

    @property
    def pixel_scale(self):
        """The pixel scale of the FITS images."""
        return self.cts_image.pixel_scale

    @property
    def exposure_unit(self):
        """The unit of the exposure map."""
        return self.exp_image.unit


class ObservationList(object):
    """
    The class that contains all individual observations of an object that from
    the same instrument.

    Parameters
    ----------
    obs_list : list, optional
        The list of observations to be combined. The default obs_list = None.

    Notes
    -----
    All individual observations must have the same pixel scale and exposure
    map unit.
    """

    _observations = []
    """The list of all individual observations."""

    _exposure_unit = u.s
    """The unit of the exposure map."""

    def __init__(self, obs_list=None):
        if obs_list is None:
            obs_list = []
        self.observations = obs_list

    @property
    def observations(self):
        """The list of all individual observations."""
        return self._observations

    @observations.setter
    def observations(self, obs_list):
        observations = []
        if isinstance(obs_list, (list, tuple)):
            for obs in obs_list:
                if isinstance(obs, Observation):
                    observations += [obs]
                else:
                    raise TypeError(
                        "Each component in obs_list must be an "
                        "Observation instance.")
        elif isinstance(obs_list, Observation):
            observations += [obs_list]
        else:
            raise TypeError("obs_list must be an Observation instance or "
                            "a list or a tuple")
        self._observations = observations
        if len(self._observations) > 0:
            self._exposure_unit = self.observations[0].exposure_unit
        else:
            pass

    @property
    def pixel_scale(self):
        """The pixel scale of the FITS images."""
        return self.observations[0].pixel_scale

    @property
    def exposure_unit(self):
        """The unit of the exposure map."""
        return self._exposure_unit

    @exposure_unit.setter
    def exposure_unit(self, unit):
        if isinstance(unit, str):
            self._exposure_unit = u.Unit(unit)
        elif isinstance(unit, u.Unit):
            self._exposure_unit = unit
        else:
            raise TypeError("unit must be a string or an Unit object.")

    def add_observation(self, new_observation):
        """
        Add a new observation object into the current observation list.

        Parameters
        ----------
        new_observation : Observation
            The new observation.

        """
        if isinstance(new_observation, Observation):
            self._observations += [new_observation]
            self._exposure_unit = new_observation.exposure_unit

    def add_observation_from_file(self, cts_image_file, exp_image_file,
                                  bkg_image_file, bkg_norm_type="count",
                                  bkg_norm_keyword="bkgnorm", extension=0):
        """
        Load a new observation from FITS files and add it into the current
        observation list.

        Parameters
        ----------
        cts_image_file : str
            The count image of the observation.
        exp_image_file : str
            The exposure map of the observation.
        bkg_image_file : str
            The image that represents the background level of the observation.
        bkg_norm_type : {"count", "flux"}, optional
            The type of the background scaling factor.
            "count" : the ratio between the total background count number in
            the count image and that in the background image.
            "flux" : the ratio between the total background count rate in the
            count image and that in the background image.
            For the "flux" type, the exposure time of the background image need
            to be properly provided. The default norm_type = "count".
        bkg_norm_keyword : str, optional
            The keyword of scaling factor in the background image header.
            The default norm_keyword = "bkgnorm".
        extension : int, optional
            The number of the hdu that contains the image data. The default
            extension = 0.

        Notes
        -----
        The exposure_unit attribute will be updated.

        See Also
        --------
        Observation

        """
        new_observation = Observation(cts_image_file, exp_image_file,
                                      bkg_image_file,
                                      bkg_norm_type=bkg_norm_type,
                                      bkg_norm_keyword=bkg_norm_keyword,
                                      extension=extension)
        self.add_observation(new_observation)

    def get_profile(self, region_list, channel_width=1, profile_axis="x"):
        """
        Get an averaged surface brightness profile from the current observation
        list .

        Parameters
        ----------
        region_list : RegionList
        channel_width
        profile_axis

        Returns
        -------

        """
        if not isinstance(region_list, RegionList):
            raise TypeError("region_list must be a RegionList.")
        else:
            region_list: RegionList

        raw_profile = Table(names=("r", "cts", "exp", "raw_bkg", "scaled_bkg"),
                            dtype=(float, float, float, float, float))
        for obs in self._observations:
            obs: Observation
            image_grid = obs.exp_image.data

            # calculate x and y for each pixel
            profile_x, valid_mask = \
                region_list.include.get_x_coordinate(image_grid, obs.cts_image.header,
                                                     axis=profile_axis)

            pixel_scale = get_pixel_scale(obs.cts_image.header) * 3600  # arcsec
            if profile_axis == "x":
                profile_x *= pixel_scale

            valid_mask = np.logical_and(valid_mask, image_grid > 0)

            # exclude masked regions
            for exclude_region in region_list.exclude:
                exclude_region: ExcludeRegion
                include_mask = exclude_region.mask(image_grid,
                                                   obs.cts_image.header)
                valid_mask: np.ndarray = np.logical_and(valid_mask,
                                                        include_mask)
            valid_mask_array = valid_mask.flatten()

            x_array = profile_x.flatten()[valid_mask_array]
            cts_array = obs.cts_image.data.flatten()[valid_mask_array]
            exp_array = obs.exp_image.data.flatten()[valid_mask_array]
            raw_bkg_array = obs.bkg_image.data.flatten()
            if obs.bkg_image.norm_type == "count":
                scaled_bkg_array: np.ndarray = raw_bkg_array * \
                                               obs.bkg_image.bkgnorm
            elif obs.bkg_image.norm_type == "flux":
                scaled_bkg_array: np.ndarray = \
                    raw_bkg_array * obs.bkg_image.bkgnorm \
                    * obs.cts_image.exptime / obs.bkg_image.exptime
            else:
                raise ValueError("Norm type must be {'count', 'flux'}.")
            raw_bkg_array = raw_bkg_array[valid_mask_array]
            scaled_bkg_array = scaled_bkg_array[valid_mask_array]

            sub_profile = Table([x_array, cts_array, exp_array, raw_bkg_array,
                                 scaled_bkg_array],
                                names=("r", "cts", "exp", "raw_bkg",
                                       "scaled_bkg"),
                                dtype=(float, float, float, float, float))
            raw_profile = vstack([raw_profile, sub_profile])

        raw_profile.sort("r")
        return Profile(raw_profile, pixel_scale=self.pixel_scale,
                       profile_axis=profile_axis, channel_width=channel_width,
                       exposure_unit=self.exposure_unit)

    def get_surface(self, region_list):

        if not isinstance(region_list, RegionList):
            raise TypeError("region_list must be a RegionList.")
        else:
            region_list: RegionList

        stacked_cts_image = np.zeros_like(self._observations[0].cts_image.data)
        stacked_exp_image = np.zeros_like(stacked_cts_image)
        stacked_bkg_image = np.zeros_like(stacked_cts_image)
        for obs in self._observations:
            obs: Observation
            stacked_cts_image += obs.cts_image.data
            stacked_exp_image += obs.exp_image.data

            if obs.bkg_image.norm_type == "count":
                scaled_bkg_image: np.ndarray = obs.bkg_image.data* \
                                               obs.bkg_image.bkgnorm
            elif obs.bkg_image.norm_type == "flux":
                scaled_bkg_image: np.ndarray = \
                    obs.bkg_image.data * obs.bkg_image.bkgnorm \
                    * obs.cts_image.exptime / obs.bkg_image.exptime
            else:
                raise ValueError("Norm type must be {'count', 'flux'}.")
            stacked_bkg_image += scaled_bkg_image

        header = obs.cts_image.header
        wcs = WCS(header)
        include_mask = stacked_exp_image > 0
        for exclude_region in region_list.exclude:
            exclude_region: ExcludeRegion
            include_mask = np.logical_and(include_mask, exclude_region.mask(stacked_cts_image, header))
        stacked_exp_image[np.logical_not(include_mask)] = 0

        surface = Surface(stacked_cts_image, stacked_exp_image, stacked_bkg_image, wcs)
        return surface


