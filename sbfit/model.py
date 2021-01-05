from collections import OrderedDict
import numpy as np
from scipy import integrate
from astropy.modeling import Model, Parameter
from astropy.modeling.utils import get_inputs_and_params
from numba import njit

__all__ = ["custom_model", "Constant", "DoublePowerLaw", "Beta",
           "ConeDoublePowerLaw"]


class BasicModel(Model):

    def evaluate(self, *args):
        args = list(args)
        x = args.pop(0)
        pnames = self.param_names
        for i in range(len(args)):
            args[i] = args[i][0]
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
    # attr_dict.update({"sigma": Parameter("sigma", default=0.05, fixed=True)})

    attr_dict.update({"__module__": "model",
                      "__doc__": func.__doc__,
                      "n_inputs": len(inputs),
                      "n_outputs": len(inputs[0].name),
                      "_func": staticmethod(func),
                      })

    return type(name, (BasicModel,), attr_dict)


@custom_model
def Constant(x, c=0):
    return c


@custom_model
def DoublePowerLaw(x, n=1, a1=0.1, a2=1.0, r=1.0, c=2.0):
    """
    x: radius for calculation;
    n: normalisation factor at the jump;
    a1: power law index 1;
    a2: power law index 2;
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
def ConeDoublePowerLaw(theta, n=1, a1=0, a2=1.0, b=10., c=2.0, z1=0.5, z2=1.0,
                       az=0.0, theta_max=70):
    def sb(n0, z, phi, b, c, a1, a2, az):
        n_in = lambda norm, rho, phi0, phi_b, alpha1, alphaz: norm * (
                phi0 / phi_b) ** (-alpha1) * rho ** (-alphaz)
        n_out = lambda norm, rho, phi0, phi_b, jump, alpha2, alphaz: \
            norm / jump * (phi0 / phi_b) ** (-alpha2) * rho ** (-alphaz)

        phi = np.abs(phi)
        out_bound = z * np.sqrt(
            1 - (np.cos(theta_max / 180 * np.pi) / np.cos(phi)) ** 2)
        n_in_sq = lambda l: n_in(n0, np.sqrt(z ** 2 + l ** 2),
                                 np.arccos(z * np.cos(phi) / np.sqrt(
                                     z ** 2 + l ** 2)), b, a1, az) ** 2
        n_out_sq = lambda l: n_out(n0, np.sqrt(z ** 2 + l ** 2),
                                   np.arctan(np.sqrt(
                                       z ** 2 * np.sin(phi) ** 2 + l ** 2) / (
                                                     z * np.cos(phi))), b,
                                   c,
                                   a2, az) ** 2
        if phi > b:
            return 2 * integrate.quad(n_out_sq, 1e-7, out_bound)[0]
        else:
            l_b = z * np.sqrt(np.cos(phi) ** 2 / np.cos(b) ** 2 - 1)
            return 2 * (integrate.quad(n_in_sq, 1e-7, l_b)[0] +
                        integrate.quad(n_out_sq, l_b, out_bound)[0])

    return integrate.quad(
        lambda t: t * sb(n, t, theta / 180 * np.pi, b / 180 * np.pi, c, a1, a2,
                         az), z1, z2)[0] * 2 / (z2 ** 2 - z1 ** 2)
