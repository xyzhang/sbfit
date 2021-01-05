import abc
from .utils import load_image
import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy import wcs

__all__ = ["CtsImage", "ExpImage", "BkgImage"]


class Image(object):

    def __init__(self, image, extension=0):
        self._header = None
        self._data = None
        self._exptime = None
        self._load_image(image, extension=extension)

    @property
    def header(self):
        return self._header

    @property
    def data(self):
        self._data: np.ndarray
        return self._data

    @property
    def shape(self):
        return self.data.shape

    @property
    def pixel_scale(self):
        return np.abs(float(self.header["CDELT1"]))

    def _load_image(self, image, extension=0):
        # Load image from fits file or from an HDU.
        if isinstance(image, str):
            self._header, self._data = load_image(image, extension=extension)
            self._header: fits.Header
        elif isinstance(image, [fits.PrimaryHDU, fits.ImageHDU]):
            self._header = image.header
            self._data = image.data

        '''
        # Calculate sky coordinate for each pixel.
        self.wcs = wcs.WCS(self._header)
        ysize, xsize = np.shape(self._data)
        x = np.arange(xsize)
        y = np.flip(np.arange(ysize))
        xc, yc = np.meshgrid(np.arange(xsize), np.arange(ysize))
        self.ra_mesh, self.dec_mesh = self.wcs.all_pix2world(xc, yc, 0)
        self.ra = self.ra_mesh[0]
        self.dec = self.dec_mesh.T[0]
        '''

    @abc.abstractmethod
    def _load_parameter(self):
        pass


class CtsImage(Image):

    def __init__(self, image, extension=0):
        super().__init__(image, extension=extension)
        self._load_parameter()

    @property
    def exptime(self):
        return self._exptime

    def _load_parameter(self):
        self._exptime = self._header['exposure']


class BkgImage(Image):

    def __init__(self, image, norm_keyword="bkgnorm", norm_type="count",
                 extension=0):
        """
        There are two types of normalisation:
            One is a ratio between counts rates. In this case we need
            additional information from exposure time.
            Another one is a ratio between counts numbers.

        Parameters
        ----------
        image
        norm_keyword
        norm_type : {"count", "flux"}
        extension
        """
        super().__init__(image, extension=extension)
        self._norm_keyword = norm_keyword
        self._norm_type = norm_type
        self._bkgnorm = None
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

    def __init__(self, image, extension=0):
        super().__init__(image, extension=extension)
        self._unit = None
        self._load_parameter()

    @property
    def unit(self):
        return self._unit

    def _load_parameter(self):
        try:
            self._unit = u.Unit(self._header["BUNIT"])
        except KeyError:
            self._unit = u.s
