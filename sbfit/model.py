from collections import OrderedDict
import numpy as np
from scipy import integrate
from astropy.modeling import Model, Parameter
from astropy.modeling.utils import get_inputs_and_params
from numba import njit

__all__ = ["custom_model", "Constant", "Gaussian", "DoublePowerLaw", "Beta",
           "ConeDoublePowerLaw"]


class BasicModel(Model):

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
    An alternative version of astropy.modeling.custom_model
    :param func:
    :return:
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
                      "n_inputs": len(inputs),
                      "n_outputs": len([inputs[0].name]),
                      "_func": staticmethod(func),
                      })

    return type(name, (BasicModel,), attr_dict)


@custom_model
def Constant(x, c=0):
    result = c
    return result


@custom_model
def Gaussian(x, n=1, x0=0, sigma=1):
    result = n * np.exp(-(x - x0) ** 2 / 2 / sigma ** 2)
    return result


@custom_model
def DoublePowerLaw(x, n=1, a1=0.1, a2=1.0, r=1.0, c=2.0):
    """
    x: radius for calculation;
    n: normalisation factor at the jump;
    a1: power law index 1;
    a: power law index 2;
    r: jump location;
    c: contraction factor.

    x, r have a _unit of arcsec;
    n has an arbitrary _unit.
    """
    if x < r:
        f = n ** 2 * (c ** 2 * integrate.quad(_dpl_project, 1e-10,
                                              np.sqrt(r ** 2 - x ** 2),
                                              args=(x, r, a1))[0] +
                      integrate.quad(_dpl_project, np.sqrt(r ** 2 - x ** 2),
                                     np.inf, args=(x, r, a2))[0])
    else:
        f = n ** 2 * \
            integrate.quad(_dpl_project, 1e-5, np.inf, args=(x, r, a2))[0]

    return f


# divided functions for speed-up
@njit
def _dpl_project(z, x, r, a):
    return ((x ** 2 + z ** 2) / r ** 2) ** (-2 * a)


@custom_model
def Beta(x, s, b, r, constant):
    return s * (1 + (x / r) ** 2) ** (0.5 - 3 * b) + constant


@custom_model
def ConeDoublePowerLaw(x, n=1, a1=0, a2=1.0, b=10., c=2.0, z1=0.5, z2=1.0,
                       az=0.0, theta_max=70, center=0):
    x = x - center

    result = integrate.quad(
        lambda t: t * _cone_sb(n, t, x / 180 * np.pi, b / 180 * np.pi, c,
                               a1, a2,
                               az, theta_max), z1, z2)[0] * 2 / (
                     z2 ** 2 - z1 ** 2)
    return result


@njit
def _cone_n_sq(l, n0, z, phi, b, a, az):
    rho = np.sqrt(z ** 2 + l ** 2)
    phi0 = np.arccos(z * np.cos(phi) / np.sqrt(z ** 2 + l ** 2))
    return (n0 * (phi0 / b) ** (-a) * rho ** (-az)) ** 2


def _cone_sb(n0, z, phi, b, c, a1, a2, az, theta_max):
    phi = np.abs(phi)

    out_bound = z * np.sqrt(
        1 - (np.cos(theta_max / 180 * np.pi) / np.cos(phi)) ** 2)
    if phi > b:
        return 2 * integrate.quad(_cone_n_sq, 1e-7, out_bound,
                                  args=(n0, z, phi, b, a2, az),
                                  )[0]
    else:
        l_b = z * np.sqrt(np.cos(phi) ** 2 / np.cos(b) ** 2 - 1)
        return 2 * (c ** 2 * integrate.quad(_cone_n_sq, 1e-7, l_b,
                                            args=(n0, z, phi, b, a1, az))[0] +
                    integrate.quad(_cone_n_sq, l_b, out_bound,
                                   args=(n0, z, phi, b, a2, az))[0])
