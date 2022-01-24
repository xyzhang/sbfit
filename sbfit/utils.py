import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from astropy.io import fits
from astropy.modeling import Model, Fittable1DModel, Parameter
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from vorbin.voronoi_2d_binning import voronoi_2d_binning


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


def get_uncertainty(sample, nbins=None, centroid=None):
    """
    Calculate the mode and 1sigma uncertainties for a given discrete sample.

    Parameters
    ----------
    sample : Array
    nbins : int
        Number of histogram bins. Default = None.
    centroid : number
        The given mode. Default = None.

    Returns
    -------
    mode : float
    up_error : float
    low_error : float

    """

    # clip Nan values
    sample = sample[np.logical_not(np.isnan(sample))]

    if nbins is None:
        bin_width = np.diff(np.percentile(sample, [16, 84]))[0] / 10
        bin_number = int((np.max(sample) - np.min(sample)) / bin_width)
        bin_number = int(len(sample) / 100)
    else:
        bin_number = nbins

    norm, grid = np.histogram(sample, bins=bin_number)
    grid_center = 0.5 * (grid[:-1] + grid[1:])

    mode_index = np.argmax(norm)
    mode_norm = norm[mode_index]
    mode = grid_center[mode_index]

    def frac(height, norm_list, show_index=False):
        low_index = np.where(norm_list > height)[0][0]
        up_index = np.where(norm_list > height)[0][-1]
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

    if centroid is None:
        pass
    else:
        mode = centroid

    low_error = mode - grid_center[low]
    up_error = grid_center[up] - mode

    return mode, up_error, low_error


def write_fits_image(array, outfile, header=None):
    """
    Write an array (image) to a FITS file.

    Parameters
    ----------
    array : numpy.ndarray
        Input array
    outfile : str
        Output destination
    header : astropy.io.fits.Header, optional
        FITS header to output
    """
    hdu = fits.HDUList()
    hdu.append(fits.ImageHDU(array, header=header))
    hdu.writeto(outfile, overwrite=True)


def voronoi(image, xmin, xmax, ymin, ymax, snr=None, pixsize=1):
    """
    Bin the image using voronoi tessellation.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array.
    xmin : int
        Lower limit of X.
    xmax : int
        Upper limit of X.
    ymin : int
        Lower limit of Y.
    ymax : int
        Upper limit of Y.
    snr : number, optional
        S/N ratio for binning. If set to None, it will be automatically estimated. Default = None

    Returns
    -------
    index_array : np.ndarray
        A 2D array of bin number


    """
    image: np.ndarray
    image[np.where(image < 0)] = 0

    # set boundary
    ly, lx = image.shape
    xmin = np.max([0, xmin])
    xmax = np.min([lx - 1, xmax])
    ymin = np.max([0, ymin])
    ymax = np.min([ly - 1, ymax])

    # set grid
    xcoor, ycoor = np.meshgrid(np.arange(xmin, xmax + 1),
                               np.arange(ymin, ymax + 1))
    filtered_image = image[ymin:ymax + 1, xmin:xmax + 1]
    xcoor1d = np.ravel(xcoor)
    ycoor1d = np.ravel(ycoor)
    signal = np.ravel(filtered_image)

    if snr is None:
        snr = np.int(np.sqrt(np.sum(filtered_image) / 100))
    else:
        pass
    print(f"Voronoi binning S/N = {snr}.")

    # binning
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = \
        voronoi_2d_binning(xcoor1d, ycoor1d, signal, np.sqrt(signal) + 1e-2, snr,
                           plot=True, pixelsize=pixsize, quiet=True, )
    index_array = np.zeros_like(filtered_image)
    index_array[ycoor1d - ymin, xcoor1d - xmin] = binNum
    plt.show()
    return index_array, xcoor, ycoor


def stat_with_index_2d(input, xcoor, ycoor, bin_number, method="sum"):
    filtered = input[ycoor, xcoor]
    bin_number = bin_number.astype(int)
    statistic, _, _ = stats.binned_statistic(np.ravel(bin_number), np.ravel(filtered),
                                             statistic=method,
                                             bins=np.arange(np.max(bin_number) + 1), )
    return statistic
