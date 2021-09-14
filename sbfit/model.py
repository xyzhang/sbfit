"""
This module defines classes of surface brightness profile models.
"""

from collections import OrderedDict

import numpy as np
from scipy import integrate
from astropy.modeling import Model, Parameter
from astropy.modeling.utils import get_inputs_and_params
from astropy.cosmology import FlatLambdaCDM
from astropy import units
from numba import njit

__all__ = ["custom_model", "Constant", "Gaussian", "DoublePowerLaw", "Beta",
           "ConeDoublePowerLaw"]

cosmos = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.7)


class BasicModel(Model):
    """
    The base class of all model classes, based on astropy.modelling.Model.
    """

    def evaluate(self, *args):
        args = list(args)
        x = args.pop(0)
        pnames = self.param_names
        for i in range(len(args)):
            args[i] = args[i]
        kwargs = dict(zip(pnames, args))
        if isinstance(x, np.ndarray):
            y = [self._func(x[i], **kwargs) for i in range(len(x))]
            return np.array(y)
        elif isinstance(x, (int, float)):
            return self._func(x, **kwargs)


def custom_model(func):
    """
    Model class decorator.
    An alternative version of astropy.modeling.custom_model.


    """
    return __model_wrapper(func)


def __model_wrapper(func):
    name = func.__name__
    attr_dict = OrderedDict()

    # include parameters
    inputs, params = get_inputs_and_params(func)
    for para in params:
        attr_dict.update(
            {para.name: Parameter(para.name, default=para.default)})

    attr_dict.update({"__module__": "model",
                      "__doc__": func.__doc__,
                      "n_inputs": 1,
                      "n_outputs": 1,
                      "_func": staticmethod(func),
                      })

    return type(name, (BasicModel,), attr_dict)


@custom_model
def Constant(x, norm=0.):
    """
    Constant model.

    Parameters
    ----------
    x : number or np.ndarray
        The input number for calculation.
    norm : float
        The output constant.

    """
    return norm


@custom_model
def Gaussian(x, norm=1, x0=0, sigma=1):
    """
    Gaussian profile model.

    Parameters
    ----------
    x : number or np.ndarray
        The input number for calculation.
    norm : number
        The normalization at the peak.
    x0 : number
        The center of the Gaussian profile.
    sigma : number
        The width of the profile.

    """
    result = norm * np.exp(-(x - x0) ** 2 / 2 / sigma ** 2)
    return result


@custom_model
def DoublePowerLaw(x, norm=1, a1=0.1, a2=1.0, r=1.0, c=2.0):
    """
    Projected double power law profile.

    Parameters
    ----------
    x : number
        The input number for calculation.
    norm : number
        The density normalization at the jump.
    a1 : number
        The power law index of the density profile before the jump.
    a2 : number
        The power law index of the density profile after the jump.
    r : number
        The jump location.
    c : number
        The strength of the jump, i.e., the contraction factor.

    Notes
    ----------
    x, r have a unit of arcsec;
    n has an arbitrary unit.

    References
    ----------
    Owers et al. 2009, ApJ, 704, 1349.

    """
    if x < r:
        result = norm ** 2 * (c ** 2 * integrate.quad(_dpl_project, 1e-10,
                                                      np.sqrt(r ** 2 - x ** 2),
                                                      args=(x, r, a1))[0] +
                              integrate.quad(_dpl_project,
                                             np.sqrt(r ** 2 - x ** 2),
                                             np.inf, args=(x, r, a2))[0])
    else:
        result = norm ** 2 * \
                 integrate.quad(_dpl_project, 1e-5, np.inf, args=(x, r, a2))[0]

    return result


# divided functions for speed-up
@njit
def _dpl_project(z, x, r, a):
    return ((x ** 2 + z ** 2) / r ** 2) ** (-2 * a)


@custom_model
def Beta(x, norm=1., beta=1., r=1.):
    """
    Beta profile.

    Parameters
    ----------
    x : number
        The input number for calculation.
    norm : number
        The normalization at the center of the cluster.
    beta : number
        The beta parameter.
    r : number
        The core radius.

    References
    ----------
    Cavaliere, A. & Fusco-Femiano, R. 1976, A&A, 500, 95
    """
    result = norm * (1 + (x / r) ** 2) ** (0.5 - 3 * beta)
    return result


@custom_model
def ConeDoublePowerLaw(x, norm=1, a1=0, a2=1.0, phi_b=10., c=2.0, r1=0.5,
                       r2=1.0, phi_max=70, center=0, redshift=0.1):
    """
    A projected cone shaped double power law profile.

    Parameters
    ----------
    x : number
        The input number for calculation.
    norm : number
        The electron density after the break (cm^-3).
    a1 : number
        The first power law index.
    a2 : number
        The second power law index
    phi_b : number
        Break position.
    c : number
        Strength of discontinuity.
    r1 : float
        Inner radius of the sector (arcsec).
    r2 : float
        Outer radius of the sector (arcsec).
    phi_max : number
        The outer azimuth boundary.
    center : number
        The center position of the cone.
    redshift: float
        redshift.

    """
    x = x - center

    distance = cosmos.luminosity_distance(redshift).to(units.kpc).value

    r1 = r1 / 3600 / 180 * np.pi * distance * 3.09e21
    r2 = r2 / 3600 / 180 * np.pi * distance * 3.09e21

    # em = integrate.quad(_cone_sb, r1, r2,
    em = integrate.quad(_r_cone_sb, r1, r2,
                        args=(norm, x / 180 * np.pi, phi_b / 180 * np.pi,
                              c, a1, a2, phi_max))[0]
    # result = em * (r1 + r2) / (r2 ** 2 - r1 ** 2)
    result = em * 2 / (r2 ** 2 - r1 ** 2)
    cf = 4e-15  # (ph s^-1 cm^3) 0.5-2.0 keV cooling function of a 5 keV plasma
    sq_arcsec_per_sq_radian = 8.46e-8
    result *= cf / (4 * np.pi) / (1 + redshift) ** 4 * sq_arcsec_per_sq_radian

    return result


@njit
def _cone_n_sq(l, n0, z, phi, b, a):
    rho = np.sqrt(z ** 2 + l ** 2)
    phi0 = np.arccos(z * np.cos(phi) / np.sqrt(z ** 2 + l ** 2))
    return (n0 * (phi0 / b) ** (-a)) ** 2


def _cone_sb(z, n0, theta, b, c, a1, a2, phi_max):
    """

    Parameters
    ----------
    z : number
    n0 : number
    theta : number
    b : number
    c : number
    a1 : number
    a2 : number
    az : number
    phi_max : number

    Returns
    -------

    """
    theta = np.abs(theta)

    out_bound = z * np.sqrt(
        1 - (np.cos(phi_max / 180 * np.pi) / np.cos(theta)) ** 2)
    if theta > b:
        result = 2 * integrate.quad(_cone_n_sq, 0, out_bound,
                                    args=(n0 / c, z, theta, b, a2),
                                    )[0]
    else:
        l_b = z * np.sqrt(np.cos(theta) ** 2 / np.cos(b) ** 2 - 1)
        result = 2 * (
                integrate.quad(_cone_n_sq, 0, l_b,
                               args=(n0, z, theta, b, a1))[0] +
                integrate.quad(_cone_n_sq, l_b, out_bound,
                               args=(n0 / c, z, theta, b, a2))[0])

    return result


def _r_cone_sb(z, n0, theta, b, c, a1, a2, phi_max):
    result = z * _cone_sb(z, n0, theta, b, c, a1, a2, phi_max)
    return result
