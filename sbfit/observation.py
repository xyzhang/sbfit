import numpy as np
from astropy.table import Table, vstack

from .image import CtsImage, ExpImage, BkgImage
from .profile import Profile
from .region import *

__all__ = ["Observation", "ObservationList"]


class Observation(object):

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
        return self._cts_image

    @property
    def exp_image(self):
        return self._exp_image

    @property
    def bkg_image(self):
        return self._bkg_image

    @property
    def telescope(self):
        try:
            telescope = self._cts_image.header["TELESCOP"]
        except KeyError:
            telescope = None
        return telescope

    @property
    def instrument(self):
        try:
            instrument = self._cts_image.header["INSTRUME"]
        except KeyError:
            instrument = None
        return instrument

    @property
    def pixel_scale(self):
        return self.cts_image.pixel_scale


class ObservationList(object):

    def __init__(self, obs_list):
        self.observations = obs_list

    @property
    def observations(self):
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
                        "Each component in obs_list must be an \
                        Observation instance.")
        elif isinstance(obs_list, Observation):
            observations += [obs_list]
        else:
            raise TypeError("obs_list must be an Observation instance or \
            a list or a tuple")
        self._observations = observations

    @property
    def pixel_scale(self):
        return self.observations[0].pixel_scale

    def get_profile(self, region_list, channel_width=1, profile_axis="x"):
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
                region_list.include.get_xy(image_grid, obs.cts_image.header,
                                           axis=profile_axis)

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
                       profile_axis=profile_axis, channel_width=channel_width)
