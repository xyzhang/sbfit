"""
This module defines classes of images that read from FITS files.
"""

import abc

import numpy as np
from astropy.io import fits
import astropy.units as u

from .utils import load_image

__all__ = ["CtsImage", "ExpImage", "BkgImage"]


class Image(object):
    """
    Base class of all image classes.
    """

    _header = None
    """The header of the FITS image."""

    _data = None
    """The data of the FITS image."""

    _exptime = None
    """The exposure time of the FITS image."""

    def __init__(self, image, extension=0):
        self._load_image(image, extension=extension)

    @property
    def header(self):
        """
        The header of the FITS image.
        """
        return self._header

    @property
    def data(self):
        """
        The data of the FITS image.
        """
        self._data: np.ndarray
        return self._data

    @property
    def shape(self):
        """
        The shape of the image array.

        Notes
        -----
        For an image with a size of m * n, the size of this array is (n, m).
        """
        return self.data.shape

    @property
    def pixel_scale(self):
        """
        The pixel scale of the FITS image.

        Notes
        -----
        pixel_scale is in a unit of degree / pixel.
        """
        return np.abs(float(self.header["CDELT1"]))

    def _load_image(self, image, extension=0):
        """
        Load image from a FITS file or an HDU.

        Parameters
        ----------
        image : str or HDU
            The image to be loaded.
        extension : int, optional
            The number of the hdu that contains the image data. The default
            extension = 0.

        """
        if isinstance(image, str):
            self._header, self._data = load_image(image, extension=extension)
            self._header: fits.Header
        elif isinstance(image, (fits.PrimaryHDU, fits.ImageHDU)):
            self._header = image.header
            self._data = image.data

    @abc.abstractmethod
    def _load_parameter(self):
        """
        The abstract method to load parameters while initialization.
        """


class CtsImage(Image):
    """
    The count image class.

    Parameters
    ----------
    image : str or HDU
        The image to be loaded.
    extension : int, optional
        The number of the hdu that contains the image data. The default
        extension = 0.
    """

    def __init__(self, image, extension=0):
        super().__init__(image, extension=extension)
        self._load_parameter()

    @property
    def exptime(self):
        return self._exptime

    def _load_parameter(self):
        self._exptime = self._header['exposure']


class BkgImage(Image):
    """

    Parameters
    ----------
    image : str or HDU
        The image to be loaded.
    norm_keyword : str, optional
        The keyword of scaling factor in the header. The default
        norm_keyword = "bkgnorm".
    norm_type : {"count", "flux"}, optional
        The type of the background scaling factor.
        "count" : the ratio between the total background count number in the
        count image and that in the background image.
        "flux" : the ratio between the total background count rate in the count
        image and that in the background image.
        For the "flux" type, the exposure time of the background image need to
        be properly provided. The default norm_type = "count".
    extension : int, optional
        The number of the hdu that contains the image data. The default
        extension = 0.

    Notes
    -----
    There are two types of normalisation:
        One is a ratio between counts rates. In this case we need
        additional information from exposure time.
        Another one is a ratio between counts numbers.

    """
    _norm_keyword = None
    """The keyword of scaling factor in the header."""

    _norm_type = None
    """The type of the background scaling factor."""

    _bkgnorm = 1.
    """The background scaling factor."""

    def __init__(self, image, norm_keyword="bkgnorm", norm_type="count",
                 extension=0):
        super().__init__(image, extension=extension)
        self._norm_keyword = norm_keyword
        self._norm_type = norm_type
        self._load_parameter()

    @property
    def exptime(self):
        return self._exptime

    @property
    def bkgnorm(self):
        return self._bkgnorm

    @property
    def norm_type(self):
        return self._norm_type

    def _load_parameter(self):
        self._exptime = self._header['exposure']
        try:
            self._bkgnorm = float(self._header[self._norm_keyword])
        except KeyError:
            self._bkgnorm = 1.0


class ExpImage(Image):
    """
    The exposure map class.

    Parameters
    ----------
    image : str or HDU
        The image to be loaded.
    extension : int, optional
        The number of the hdu that contains the image data. The default
        extension = 0.

    """

    _unit = None
    """The unit of the exposure map."""

    def __init__(self, image, extension=0):
        super().__init__(image, extension=extension)
        self._load_parameter()

    @property
    def unit(self):
        """The unit of the exposure map."""
        return self._unit

    def _load_parameter(self):
        try:
            self._unit = u.Unit(self._header["BUNIT"])
        except KeyError:
            self._unit = u.s
