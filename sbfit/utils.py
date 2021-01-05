import numpy as np
from astropy.io import fits
from astropy.modeling import Model
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from .exception import *


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


def get_pixel_scale(header):
    wcs = WCS(header)
    pixel_scale = np.abs(np.diag(wcs.pixel_scale_matrix)[0])
    return pixel_scale


def xy2elliptic(x, y, x0, y0, major, minor, angle, startangle, stopangle):
    """
    Here the elliptic coordinate system is just an asymmetric Cartesian system.
    The returned r will be scaled to the major axis.
    Input parameter angle has the _unit of degree.
    """
    average_angle = (startangle + stopangle) / 2
    # if major < minor:
    #    raise InvalidValueError("Minor axis should not be longer than major axis.")

    # Shift the center slightly to avoid nan.
    x0 += 1.2345678e-8
    y0 += 1.2345678e-8
    c = np.sqrt(major ** 2 - minor ** 2)
    mask_2 = np.logical_and(x - x0 >= 0, y - y0 < 0)
    mask_3 = np.logical_and(x - x0 < 0, y - y0 < 0)
    mask_4 = np.logical_and(x - x0 < 0, y - y0 >= 0)
    azimuth = np.arctan((y - y0) / (x - x0)) + np.pi * (
            mask_2 * 2. + mask_3 * 1. + mask_4 * 1.)  # from 0 to 2 pi rad.
    r_az = np.sqrt(minor ** 2 + c ** 2 / (1 + np.tan(
        azimuth - angle / 180. * np.pi) ** 2 * major ** 2 / minor ** 2))
    r_angle = np.sqrt(minor ** 2 + c ** 2 / (1 + np.tan(
        average_angle / 180. * np.pi) ** 2 * major ** 2 / minor ** 2))
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2) / r_az * r_angle

    return azimuth / np.pi * 180, r


def xyrot(x, y, xstart, ystart, xend, yend):
    xstart += 1e-8
    ystart += 1e-8
    if xend >= xstart and yend >= ystart:
        phi = np.arctan((yend - ystart) / (xend - xstart))
    elif xend < xstart and yend >= ystart:
        phi = (np.pi + np.arctan((yend - ystart) / (xend - xstart)))
    elif xend < xstart and yend < ystart:
        phi = (np.pi + np.arctan((yend - ystart) / (xend - xstart)))
    else:
        phi = (2 * np.pi + np.arctan((yend - ystart) / (xend - xstart)))

    rx = (x - xstart) * np.cos(phi) + (y - ystart) * np.sin(phi)
    ry = - (x - xstart) * np.sin(phi) + (y - ystart) * np.cos(phi)
    return rx, ry


def isincircle(x, y, x0, y0, r):
    x0 += 1.2345678e-8
    y0 += 1.2345678e-8
    distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return distance < r


def isinellipse(x, y, x0, y0, major, minor, pa):
    x0 += 1.2345678e-8
    y0 += 1.2345678e-8
    _, r = xy2elliptic(x, y, x0, y0, major, minor, pa, pa - 90, pa + 90)
    return r < major


def weighted_average(values, errors):
    mean = np.sum(values / errors ** 2) / np.sum(1 / errors ** 2)
    error = np.sqrt(1 / np.sum(errors ** -2))
    return mean, error


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
