import numpy as np
from astropy.io import fits
from astropy.modeling import Model, Fittable1DModel, Parameter
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord


def load_image(filename, extension=0):
    """
    Read FITS image
    The default extension = 0 is the primary HDU.
    """
    hdu = fits.open(filename)
    header = hdu[extension].header
    data = hdu[extension].data * 1.  # Convert to float type.
    hdu.close()
    return header, data


def sky_to_pixel(x, y, header, unit="deg", frame="icrs"):
    wcs = WCS(header)
    coord = SkyCoord(x, y, unit=unit, frame=frame)
    x_pix, y_pix = wcs.world_to_pixel(coord)
    x_pix = float(x_pix)
    y_pix = float(y_pix)
    return x_pix, y_pix


def get_free_parameter(model: Model):
    pnames_free = list(model.param_names)
    pvalues_free = list(model.parameters)
    pfixed: dict = model.fixed
    for key in pfixed.keys():
        if pfixed[key]:
            keyindex = pnames_free.index(key)
            pnames_free.pop(keyindex)
            pvalues_free.pop(keyindex)
    return pnames_free, pvalues_free


def get_pixel_scale(header):
    wcs = WCS(header)
    pixel_scale = np.abs(np.diag(wcs.pixel_scale_matrix)[0])
    return pixel_scale


def get_parameter_bounds(model: Model, param_names):
    low_bounds = []
    up_bounds = []
    for item in param_names:
        item_bounds = list(model.bounds[item])
        if item_bounds[0] is None:
            low_bounds += [-np.inf]
        else:
            low_bounds += [item_bounds[0]]
        if item_bounds[1] is None:
            up_bounds += [np.inf]
        else:
            up_bounds += [item_bounds[1]]
    return np.array(low_bounds), np.array(up_bounds)


class King(Fittable1DModel):
    n_inputs = 1
    n_outputs = 1

    amplitude = Parameter(default=1)
    rc = Parameter(default=1)
    r0 = Parameter(default=0)
    alpha = Parameter(default=1.5)

    def evaluate(self, x, amplitude, rc, r0, alpha):
        return amplitude * (1 + (x - r0) ** 2 / rc ** 2) ** -alpha
