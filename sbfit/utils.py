import numpy as np
from scipy import optimize
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


def get_uncertainty(sample):
    """
    Calculate the mode and 1sigma uncertainties for a given discrete sample.

    Parameters
    ----------
    sample : Array

    Returns
    -------

    """

    bin_width = np.diff(np.percentile(sample, [16, 84]))[0] / 10
    bin_number = int((np.max(sample) - np.min(sample)) / bin_width)
    norm, grid = np.histogram(sample, bins=bin_number)
    grid_center = 0.5 * (grid[:-1] + grid[1:])
    print(grid)

    mode_index = np.argmax(norm)
    mode_norm = norm[mode_index]
    mode = grid_center[mode_index]

    def frac(height, norm_list, show_index=False):
        low_index = np.where(norm_list > height)[0][0] - 1
        up_index = np.where(norm_list > height)[0][-1] + 1
        out_fraction = (np.sum(norm_list[:low_index]) + np.sum(
            norm_list[up_index:])) / np.sum(norm_list)
        if show_index:
            return out_fraction, low_index, up_index
        else:
            return out_fraction

    i = 1
    while frac(mode_norm - i, norm) > .318:
        i += 1
    else:
        _, low, up = frac(mode_norm - i, norm, show_index=True)

    low_error = mode - grid_center[low]
    up_error = grid_center[up] - mode

    return mode, up_error, low_error



