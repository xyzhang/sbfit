"""
This module defines classes of surface brightness profile models.
"""

from collections import OrderedDict

import numpy as np
from scipy import integrate
from astropy.modeling import Model, Parameter
from astropy.modeling.utils import get_inputs_and_params
from numba import njit

__all__ = ["custom_model", "Constant", "Gaussian", "DoublePowerLaw", "Beta",
           "ConeDoublePowerLaw"]


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
def ConeDoublePowerLaw(x, norm=1, a1=0, a2=1.0, phi_b=10., c=2.0, z1=0.5,
                       z2=1.0, phi_max=70, center=0, distance=1e5):
    """

    Parameters
    ----------
    x
    norm : number
        The electron density after the break (cm^-3).
    a1
    a2
    phi_b
    c
    z1 : float
        Inner radius of the sector (arcsec).
    z2 : float
        Outer radius of the sector (arcsec).
    phi_max
    center
    distance : float
        Distance (kpc).

    Returns
    -------

    """
    x = x - center

    z1 = z1 / 3600 / 180 * np.pi * distance * 3.09e21
    z2 = z2 / 3600 / 180 * np.pi * distance * 3.09e21

    # em = integrate.quad(_cone_sb, z1, z2,
    em = integrate.quad(_r_cone_sb, z1, z2,
                        args=(norm, x / 180 * np.pi, phi_b / 180 * np.pi,
                              c, a1, a2, phi_max))[0]
    # result = em * (z1 + z2) / (z2 ** 2 - z1 ** 2)
    result = em * 2 / (z2 ** 2 - z1 ** 2)
    result *= 3.5e-15 / (4 * np.pi) * 8.46e-8

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
