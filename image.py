from .utils import load_image
import numpy as np
from astropy.io import fits
from astropy import wcs


class Image(object):

    def __init__(self, image):
        self._load_image(image)

    def _load_image(self, image):
        # Load image from fits file or from an HDU.
        if isinstance(image, str):
            self.header, self.data = load_image(image)
            assert isinstance(self.header, fits.header.Header)
        elif isinstance(image, fits.PrimaryHDU) or isinstance(image, fits.ImageHDU):
            self.header = image.header
            self.data = image.data
        self.wcs = wcs.WCS(self.header)

        # Calculate sky coordinate for each pixel.
        xsize, ysize = np.shape(self.data)
        xc, yc = np.meshgrid(np.arange(xsize), np.arange(ysize))
        self.ra_mesh, self.dec_mesh = self.wcs.all_pix2world(xc, yc, 0)
        self.ra = self.ra_mesh[0]
        self.dec = self.dec_mesh.T[0]


class CtsImage(Image):

    def __init__(self, image):
        self._load_image(image)
        self._load_parameter()

    def _load_parameter(self):
        self.exptime = self.header['exposure']


class BkgImage(Image):

    def __init__(self, image, norm="bkgnorm", norm_type="count"):
        """
        There are two types of normalisation:
            One is a ratio between counts rates. In this case we need additional information from exposure time.
            Another one is a ratio bewteen counts numbers.
        """
        self._load_image(image)
        self.norm_keyword = norm
        self.norm_type = norm_type
        self._load_parameter()

    def _load_parameter(self):
        self.exptime = self.header['exposure']
        try:
            self.bkgnorm = float(self.header[self.norm_keyword])
        except KeyError:
            self.bkgnorm = 1.0


class ExpImage(Image):

    def __init__(self, image):
        self._load_image(image)
        self.unit = "s"
        try:
            self.unit = self.header["BUNIT"]
        except:
            pass


class ImageList(object):
    itype = None

    def __init__(self, *image_list):
        self.images = []
        self.data = []
        self.headers = []
        self.wcses = []
        self._load_data(*image_list)

    def _load_data(self, *image_list):
        for image in image_list:
            if self.itype == "counts":
                self.images.append(CtsImage(image))
            elif self.itype == "exposure":
                self.images.append(ExpImage(image))
            elif self.itype == "background":
                self.images.append(BkgImage(image, norm=self.norm_keyword, norm_type=self.norm_type))
        for image in self.images:
            self.data.append(image.data)
            self.headers.append(image.header)
            self.wcses.append(image.wcs)


# TODO Test super call.
class CtsImageList(ImageList):
    itype = "counts"

    def __init__(self, *image_list):
        super().__init__(*image_list)
        self.exptime = []
        for image in self.images:
            self.exptime += [image.exptime]


class ExpImageList(ImageList):
    itype = "exposure"

    def __init__(self, *image_list):
        super().__init__(*image_list)
        self.unit = []
        for image in self.images:
            self.unit += [image.unit]


class BkgImageList(ImageList):
    itype = "background"

    def __init__(self, *image_list, norm="bkgnorm", norm_type="count"):
        self.norm_keyword = norm
        self.norm_type = norm_type
        super().__init__(*image_list)
        self.bkgnorm = []
        self.exptime = []
        for image in self.images:
            self.bkgnorm += [image.bkgnorm]
            self.exptime += [image.exptime]
