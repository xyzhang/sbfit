"""
This module defines all supported DS9 region classes.

Regions are classified as inclusive regions and exclusive regions. Inclusive
regions are used for defining the profile extraction region shape and
direction, while exclusive regions are used to mask out contaminated pixels.
Inclusive regions are "Panda", "Epanda", and "Projection". Exclusive regions
are "Circle" and "Ellipse".

Currently, the supported coordinate frames are "fk5", "icrs", and "image".

"""

import abc

import pyregion
import numpy as np
from astropy.io import fits

from . import utils

__all__ = ["read_region", "RegionList", "IncludeRegion", "ExcludeRegion",
           "Circle", "Ellipse",
           "Epanda", "Panda", "Projection"]

VALID_FRAME = ["fk5", "icrs", "image"]


def read_region(region_file):
    """
    Read a DS9 region file and create a region list.

    Parameters
    ----------
    region_file : str
        The DS9 region file.

    Returns
    -------
    region_list : RegionList
        The region list contains all supported regions that in the region file.

    """
    pyregion_list = pyregion.open(region_file)
    region_list = RegionList()
    region_list.frame = pyregion_list[0].coord_format
    region_class = {"circle": Circle,
                    "ellipse": Ellipse,
                    # "box": Box, # TODO box
                    "projection": Projection,
                    "panda": Panda,
                    "epanda": Epanda}
    region_parameter = {"circle": ("x", "y", "radius"),
                        "ellipse": ("x", "y", "major", "minor", "angle"),
                        # "box": , # TODO box
                        "projection": ("x1", "y1", "x2", "y2", "width"),
                        "panda": ("x", "y", "startangle", "stopangle",
                                  "nangle", "inner", "outer", "nradius",
                                  "angle"),
                        "epanda": ("x", "y", "startangle", "stopangle",
                                   "nangle", "innermajor", "innerminor",
                                   "outermajor", "outerminor", "nradius",
                                   "angle")
                        }
    for item in pyregion_list:
        item: pyregion.Shape
        if item.name in region_class:
            region_obj = region_class[item.name]()
            region_obj.set_parameters(**dict(zip(region_parameter[item.name],
                                                 item.coord_list)))
            region_obj.frame = region_list.frame
            region_list.add_region(region_obj)
    return region_list


class Region(object):
    """
    The base class of all region classes.
    """

    _parameters = {}
    """The parameter dictionary."""

    _frame = "fk5"
    """The coordinate frame of the region."""

    @property
    def frame(self):
        """The coordinate frame of the region."""
        return self._frame

    @frame.setter
    def frame(self, coord_type):
        """The coordinate frame of the region."""
        if coord_type in VALID_FRAME:
            self._frame = coord_type
        else:
            raise TypeError(f"Frame type {coord_type} is invalid.")

    @property
    def parameters(self):
        """The parameter dictionary."""
        return self._parameters

    def set_parameters(self, **kwargs):
        """
        Update parameters for the region.

        Parameters
        ----------
        kwargs : region parameters
            See [1] for definitions.

        References
        ----------
        [1] http://ds9.si.edu/doc/ref/region.html

        """
        for key in kwargs:
            if key not in self._parameters:
                raise KeyError(f'"{key}" is not a region parameter.')
            else:
                pass
        self._parameters.update(kwargs)

    @staticmethod
    def _create_coord(imgdata):
        """
        Create X, Y mesh grids with the size of the image data.

        Parameters
        ----------
        imgdata : ndarray
            The image data.

        Returns
        -------
        xcoor : ndarray
            The mesh grid of X coordinate.
        ycoor : ndarray
            The mesh grid of Y coordinate.

        """
        sy, sx = imgdata.shape

        xcoor, ycoor = np.meshgrid(np.arange(sx),
                                   np.flip(np.arange(sy)),
                                   # to meet the fits coordinate definition.
                                   )

        return xcoor, ycoor


class IncludeRegion(Region):
    """
    The base class of all inclusive regions.
    """

    def get_x_coordinate(self, imgdata, header=None, axis="x"):
        """
        For a given image, calculate the X coordinate on the profile for
        each pixel. Meanwhile, an inclusive mask based on the region shape is
        calculated.

        Parameters
        ----------
        imgdata : ndarray
            The input image array to calculate X, Y coordinate based on the
            extraction region.
        header : fits.Header
            The FITS header of the input image, which is used for convert
            sky coordinate to the image coordinate.
        axis : {"x", "y"}, optional
            The extracted profile direction.
            "X": azimuthally averaged radial profile.
            "Y": radially averaged azimuthal profile (only supported by Panda).

        Returns
        -------
        profile_xcoor : ndarray
            Calculated X coordinate of each pixel.
        profile_mask : ndarray
            The include mask of the region.

        """
        img_xcoor, img_ycoor = self._create_coord(imgdata)
        profile_xcoor, profile_mask = self._get_x_coordinate_func(img_xcoor, img_ycoor,
                                                                  header, axis)
        return np.flip(profile_xcoor, axis=0), np.flip(profile_mask, axis=0)

    @abc.abstractmethod
    def _get_x_coordinate_func(self, img_xcoor, img_ycoor, header, axis):
        """
        An abstract method for each inclusive region class, which is defined
        for calculating the X coordinate of each pixel in a profile.
        """


class ExcludeRegion(Region):
    """
    The base class of all exclusive regions.
    """

    def mask(self, imgdata, header=None):
        """

        Parameters
        ----------
        imgdata : ndarray
            The input image array based on which the exclusion mask is defined.
        header : fits.Header
            The FITS header of the input image, which is used for convert
            sky coordinate to the image coordinate.

        Returns
        -------
        mask : ndarray
            The calculated mask.

        """
        img_xcoor, img_ycoor = self._create_coord(imgdata)
        mask = self._mask_func(img_xcoor, img_ycoor, header)
        return np.flip(mask, axis=0)

    @abc.abstractmethod
    def _mask_func(self, img_xcoor, img_ycoor, header):
        """
        An abstract method for each exclusive region class, which is defined
        for calculating the exclusive mask.
        """
        pass


class Circle(ExcludeRegion):
    """
    The Circle region.
    """

    def __init__(self):
        self._parameters = {"x": None,
                            "y": None,
                            "radius": None}

    def _mask_func(self, img_xcoor, img_ycoor, header):
        x = self._parameters["x"]
        y = self._parameters["y"]
        radius = self._parameters["radius"]

        if self.frame in ["fk5", "icrs"]:
            x, y = utils.sky_to_pixel(x, y, header, unit="deg",
                                      frame=self.frame)
            pixel_scale = utils.get_pixel_scale(header)
            radius /= pixel_scale

        distance = np.sqrt(
            (img_xcoor - x) ** 2 + (img_ycoor - y) ** 2)
        mask = distance > radius
        return mask


class Ellipse(ExcludeRegion):
    """
    The Ellipse region.
    """

    def __init__(self):
        self._parameters = {"x": 0.,
                            "y": 0.,
                            "major": 1.,
                            "minor": 1.,
                            "angle": 0}

    def _mask_func(self, img_xcoor, img_ycoor, header):
        x = self._parameters["x"]
        y = self._parameters["y"]
        major = self._parameters["major"]
        minor = self._parameters["minor"]
        angle = self._parameters["angle"]

        if self.frame in ["fk5", "icrs"]:
            x, y = utils.sky_to_pixel(x, y, header, unit="deg",
                                      frame=self.frame)
            pixel_scale = utils.get_pixel_scale(header)
            major /= pixel_scale
            minor /= pixel_scale

        c = np.sqrt(major ** 2 - minor ** 2)
        average_angle = 0
        mask_1 = np.logical_and(img_xcoor - x >= 0, img_ycoor - y >= 0)
        mask_2 = np.logical_and(img_xcoor - x < 0, img_ycoor - y >= 0)
        mask_3 = np.logical_and(img_xcoor - x < 0, img_ycoor - y < 0)
        mask_4 = np.logical_and(img_xcoor - x >= 0, img_ycoor - y < 0)
        azimuth_1 = np.arctan((img_ycoor - y) / (img_xcoor - x))
        azimuth_2 = azimuth_1 + np.pi
        azimuth_3 = azimuth_1 + np.pi
        azimuth_4 = azimuth_1 + 2 * np.pi
        azimuth = azimuth_1 * mask_1 + azimuth_2 * mask_2 + \
                  azimuth_3 * mask_3 + azimuth_4 * mask_4
        azimuth *= 180 / np.pi

        r_az = np.sqrt(minor ** 2 + c ** 2 / (1 + np.tan(
            (azimuth - angle) / 180. * np.pi) ** 2 * major ** 2 / minor ** 2))
        r_angle = np.sqrt(minor ** 2 + c ** 2 / (1 + np.tan(
            average_angle / 180. * np.pi) ** 2 * major ** 2 / minor ** 2))
        r = np.sqrt(
            (img_xcoor - x) ** 2 + (img_ycoor - y) ** 2) / r_az * r_angle
        mask = r > major
        return mask


class Projection(IncludeRegion):
    """
    The Projection region.
    """

    def __init__(self):
        self._parameters = {"x1": None,
                            "y1": None,
                            "x2": None,
                            "y2": None,
                            "width": None}

    def _get_x_coordinate_func(self, img_xcoor, img_ycoor, header, axis):
        if axis == "x":
            pass
        else:
            raise ValueError(
                "Projection region only supports 'x' axis profile.")
        xstart = self.parameters["x1"]
        ystart = self.parameters["y1"]
        xend = self.parameters["x2"]
        yend = self.parameters["y2"]
        width = self.parameters["width"]

        if self.frame in ["fk5", "icrs"]:
            xstart, ystart = utils.sky_to_pixel(xstart, ystart, header,
                                                unit="deg", frame=self.frame)
            xend, yend = utils.sky_to_pixel(xend, yend, header,
                                            unit="deg", frame=self.frame)
            pixel_scale = utils.get_pixel_scale(header)
            width /= pixel_scale

        if xend >= xstart and yend >= ystart:
            phi = np.arctan((yend - ystart) / (xend - xstart))
        elif xend < xstart and yend >= ystart:
            phi = (np.pi + np.arctan((yend - ystart) / (xend - xstart)))
        elif xend < xstart and yend < ystart:
            phi = (np.pi + np.arctan((yend - ystart) / (xend - xstart)))
        else:
            phi = (2 * np.pi + np.arctan((yend - ystart) / (xend - xstart)))

        x_prof = (img_xcoor - xstart) * np.cos(phi) + \
                 (img_ycoor - ystart) * np.sin(phi)
        y_prof = - (img_xcoor - xstart) * np.sin(phi) + \
                 (img_ycoor - ystart) * np.cos(phi)
        profile_mask = np.logical_and(np.logical_and(y_prof >= 0,
                                                     y_prof <= width),
                                      x_prof >= 0)
        return x_prof, profile_mask


class Epanda(IncludeRegion):
    """
    The Epanda region.
    """

    def __init__(self):
        self._parameters = {"x": 0.,
                            "y": 0.,
                            "startangle": 0.,
                            "stopangle": 180.,
                            "nangle": 1,
                            "innermajor": 0.,
                            "innerminor": 0.,
                            "outermajor": 1.,
                            "outerminor": 1.,
                            "nradius": 1,
                            "angle": 0}

    def _get_x_coordinate_func(self, img_xcoor, img_ycoor, header, axis):
        if axis == "x":
            pass
        else:
            raise ValueError(
                "Projection region only supports 'x' axis profile.")
        x = self.parameters["x"]
        y = self.parameters["y"]
        startangle = self.parameters["startangle"]
        stopangle = self.parameters["stopangle"]
        outermajor = self.parameters["outermajor"]
        outerminor = self.parameters["outerminor"]
        innermajor = self.parameters["innermajor"]
        innerminor = self.parameters["innerminor"]
        angle = self.parameters["angle"]

        if self.frame in ["fk5", "icrs"]:
            x, y = utils.sky_to_pixel(x, y, header, unit="deg",
                                      frame=self.frame)
            pixel_scale = utils.get_pixel_scale(header)
            outermajor /= pixel_scale
            outerminor /= pixel_scale
            innermajor /= pixel_scale
            innerminor /= pixel_scale

        average_angle = (startangle + stopangle) / 2
        c = np.sqrt(np.abs(outermajor ** 2 - outerminor ** 2))

        mask_1 = np.logical_and(img_xcoor - x >= 0, img_ycoor - y >= 0)
        mask_2 = np.logical_and(img_xcoor - x < 0, img_ycoor - y >= 0)
        mask_3 = np.logical_and(img_xcoor - x < 0, img_ycoor - y < 0)
        mask_4 = np.logical_and(img_xcoor - x >= 0, img_ycoor - y < 0)
        azimuth_1 = np.arctan((img_ycoor - y) / (img_xcoor - x))
        azimuth_2 = azimuth_1 + np.pi
        azimuth_3 = azimuth_1 + np.pi
        azimuth_4 = azimuth_1 + 2 * np.pi
        azimuth = azimuth_1 * mask_1 + azimuth_2 * mask_2 + \
                  azimuth_3 * mask_3 + azimuth_4 * mask_4
        azimuth *= 180 / np.pi

        r_az = np.sqrt(outerminor ** 2 + c ** 2 / (
                1 + np.tan((azimuth - angle) / 180. * np.pi) ** 2
                * outermajor ** 2 / outerminor ** 2))
        r_angle = np.sqrt(outerminor ** 2 + c ** 2 / (1 + np.tan(
            average_angle / 180. * np.pi) ** 2 * outermajor ** 2 /
                                                      outerminor ** 2))
        r = np.sqrt(
            (img_xcoor - x) ** 2 + (img_ycoor - y) ** 2) / r_az * r_angle

        # filter ranges
        startangle += angle
        stopangle += angle

        if stopangle > 360:
            profile_mask = np.logical_or(
                np.logical_and(azimuth >= 0,
                               azimuth <= stopangle - 360),
                np.logical_and(azimuth >= startangle,
                               azimuth < 360)
            )
        else:
            profile_mask = np.logical_and(azimuth >= startangle,
                                          azimuth <= stopangle)
        return r, profile_mask


class Panda(IncludeRegion):
    """
    The Panda region.
    """

    def __init__(self):
        self._parameters = {"x": 0.,
                            "y": 0.,
                            "startangle": 0.,
                            "stopangle": 180.,
                            "nangle": 1,
                            "inner": 1.,
                            "outer": 1.,
                            "nradius": 1,
                            "angle": 0}

    def _get_x_coordinate_func(self, img_xcoor, img_ycoor, header, axis):
        x = self.parameters["x"]
        y = self.parameters["y"]
        startangle = self.parameters["startangle"]
        stopangle = self.parameters["stopangle"]
        outer = self.parameters["outer"]
        inner = self.parameters["inner"]
        angle = self.parameters["angle"]
        if self.frame in ["fk5", "icrs"]:
            x, y = utils.sky_to_pixel(x, y, header, unit="deg",
                                      frame=self.frame)
            pixel_scale = utils.get_pixel_scale(header)
            inner /= pixel_scale
            outer /= pixel_scale

        mask_1 = np.logical_and(img_xcoor - x >= 0, img_ycoor - y >= 0)
        mask_2 = np.logical_and(img_xcoor - x < 0, img_ycoor - y >= 0)
        mask_3 = np.logical_and(img_xcoor - x < 0, img_ycoor - y < 0)
        mask_4 = np.logical_and(img_xcoor - x >= 0, img_ycoor - y < 0)
        azimuth_1 = np.arctan((img_ycoor - y) / (img_xcoor - x))
        azimuth_2 = azimuth_1 + np.pi
        azimuth_3 = azimuth_1 + np.pi
        azimuth_4 = azimuth_1 + 2 * np.pi
        azimuth = azimuth_1 * mask_1 + azimuth_2 * mask_2 \
                  + azimuth_3 * mask_3 + azimuth_4 * mask_4
        azimuth *= 180 / np.pi

        r = np.sqrt((img_xcoor - x) ** 2 + (img_ycoor - y) ** 2)

        # filter ranges
        startangle += angle
        stopangle += angle
        if axis == "x":

            if stopangle > 360:
                profile_mask = np.logical_or(
                    np.logical_and(azimuth >= 0,
                                   azimuth <= stopangle - 360),
                    np.logical_and(azimuth >= startangle,
                                   azimuth < 360)
                )
            else:
                profile_mask = np.logical_and(azimuth >= startangle,
                                              azimuth <= stopangle)
            return r, profile_mask
        elif axis == "y":
            profile_mask = np.logical_and(r >= inner,
                                          r <= outer)
            azimuth -= startangle
            negtive_mask = azimuth < 0
            azimuth[negtive_mask] += 360
            return azimuth, profile_mask
        else:
            raise ValueError("Axis must be 'x' or 'y'.")


class RegionList(object):
    """
    This is a container of a number of region objects. The RegionList object
    can have one inclusive region and multiple exclusive regions.

    Parameters
    ----------
    frame : {"fk5", "icrs", "image"}, optional
        The image coordinate frame of the regions. The default frame = "fk5".

    """

    _include = IncludeRegion()

    _exclude = []

    _frame = "fk5"

    def __init__(self, frame="fk5"):
        self.frame = frame

    @property
    def include(self):
        """The inclusive region."""
        return self._include

    @property
    def exclude(self):
        """The exclusive region list."""
        return self._exclude

    @property
    def frame(self):
        """The region coordinate frame."""
        return self._frame

    @frame.setter
    def frame(self, coord_type):
        if coord_type in VALID_FRAME:
            self._frame = coord_type
        else:
            raise TypeError(f"Frame type {coord_type} is invalid. Only"
                            f"{VALID_FRAME} types are supported.")

    def add_region(self, region):
        """
        Update the inclusive region or add an exclusive region into the current
        exclusive region list.

        Parameters
        ----------
        region : Region
            The region object to be updated or added.

        """
        if isinstance(region, IncludeRegion):
            self._include = region
        elif isinstance(region, ExcludeRegion):
            self._exclude += [region]
        else:
            raise TypeError(f"{type(region)} is not a region.")
