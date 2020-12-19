import copy
from collections import OrderedDict
import numpy as np
from scipy import integrate, ndimage, misc
from astropy.modeling import Model, Fittable1DModel, Parameter
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.modeling.utils import get_inputs_and_params
from numba import njit

'''
def broken_pwl(x, n, a1, a2, r, c, constant):
    """
    x: radius for calculation;
    n: normalisation factor at the jump;
    a1: power law index 1;
    a2: power law index 2;
    r: jump location;
    c: contraction factor.

    x, r have a unit of arcmin;
    n has an arbitrary unit.
    """
    if x < r:
        f = n * (integrate.quad(lambda z: c ** 2 * ((x ** 2 + z ** 2) / r ** 2) ** -a1, 1e-4,
                                np.sqrt(r ** 2 - x ** 2))[0] +
                 integrate.quad(lambda z: ((x ** 2 + z ** 2) / r ** 2) ** -a2, np.sqrt(r ** 2 - x ** 2), np.inf)[
                     0])
    else:
        f = n * integrate.quad(lambda z: ((x ** 2 + z ** 2) / r ** 2) ** -a2, 1e-4, np.inf)[0]
    return f + constant


def smoothed_broken_pwl(x, n, a1, a2, r, c, constant, sigma=0.05):
    nconvolve = int(np.round(sigma / 0.005))
    xx = np.linspace(x - 3 * sigma, x + 3 * sigma, 1 + 6 * nconvolve)
    yy = np.zeros(len(xx))
    for i in range(len(xx)):
        yy[i] = broken_pwl(xx[i], n, a1, a2, r, c, constant)
    smoothed = ndimage.gaussian_filter(yy, nconvolve)[3 * nconvolve]
    return smoothed


def derive_smoothed_broken_pwl(x, n, a1, a2, r, c, constant, sigma=0.05):
    dn = misc.derivative(lambda t: smoothed_broken_pwl(x, t, a1, a2, r, c, constant, sigma=sigma), n, dx=1e-3)
    da1 = misc.derivative(lambda t: smoothed_broken_pwl(x, n, t, a2, r, c, constant, sigma=sigma), a1, dx=1e-3)
    da2 = misc.derivative(lambda t: smoothed_broken_pwl(x, n, a1, t, r, c, constant, sigma=sigma), a2, dx=1e-3)
    dr = misc.derivative(lambda t: smoothed_broken_pwl(x, n, a1, a2, t, c, constant, sigma=sigma), r, dx=1e-3)
    dc = misc.derivative(lambda t: smoothed_broken_pwl(x, n, a1, a2, r, t, constant, sigma=sigma), c, dx=1e-3)
    dconstant = misc.derivative(lambda t: smoothed_broken_pwl(x, n, a1, a2, r, c, t, sigma=sigma), constant, dx=1e-3)
    dsigma = misc.derivative(lambda t: smoothed_broken_pwl(x, n, a1, a2, r, c, constant, sigma=t), sigma, dx=1e-3)
    return (dn, da1, da2, dr, dc, dconstant, dsigma)
'''


def beta(x, s, b, r, constant):
    return s * (1 + (x / r) ** 2) ** (0.5 - 3 * b) + constant


def smoothed_beta(x, s, b, r, constant, sigma=0.05):
    nconvolve = int(np.round(sigma / 0.005))
    xx = np.linspace(x - 3 * sigma, x + 3 * sigma, 1 + 6 * nconvolve)
    yy = np.zeros(len(xx))
    for i in range(len(xx)):
        yy[i] = beta(xx[i], s, b, r, constant)
    smoothed = ndimage.gaussian_filter(yy, nconvolve)[3 * nconvolve]
    return smoothed


def derive_smoothed_beta(x, s, b, r, constant, sigma=0.05):
    ds = misc.derivative(lambda t: smoothed_beta(x, t, b, r, constant, sigma=sigma), s, dx=1e-3)
    db = misc.derivative(lambda t: smoothed_beta(x, s, t, r, constant, sigma=sigma), b, dx=1e-3)
    dr = misc.derivative(lambda t: smoothed_beta(x, s, b, t, constant, sigma=sigma), r, dx=1e-3)
    dconstant = misc.derivative(lambda t: smoothed_beta(x, s, b, r, t, sigma=sigma), constant, dx=1e-3)
    dsigma = misc.derivative(lambda t: smoothed_beta(x, s, b, r, constant, sigma=t), sigma, dx=1e-3)
    return (ds, db, dr, dconstant, dsigma)


def smoothed_double_beta(x, s1, b1, r1, s2, b2, r2, constant, sigma=0.05):
    nconvolve = int(np.round(sigma / 0.005))
    xx = np.linspace(x - 3 * sigma, x + 3 * sigma, 1 + 6 * nconvolve)
    yy = np.zeros(len(xx))
    for i in range(len(xx)):
        yy[i] = beta(xx[i], s1, b1, r1, 0) + beta(xx[i], s2, b2, r2, constant)
    smoothed = ndimage.gaussian_filter(yy, nconvolve)[3 * nconvolve]
    return smoothed


def derive_smoothed_double_beta(x, s1, b1, r1, s2, b2, r2, constant, sigma=0.05):
    ds1 = misc.derivative(lambda t: smoothed_double_beta(x, t, b1, r1, s2, b2, r2, constant, sigma=sigma), s1, dx=1e-3)
    db1 = misc.derivative(lambda t: smoothed_double_beta(x, s1, t, r1, s2, b2, r2, constant, sigma=sigma), b1, dx=1e-3)
    dr1 = misc.derivative(lambda t: smoothed_double_beta(x, s1, b1, t, s2, b2, r2, constant, sigma=sigma), r1, dx=1e-3)
    ds2 = misc.derivative(lambda t: smoothed_double_beta(x, s1, b1, r1, t, b2, r2, constant, sigma=sigma), s2, dx=1e-3)
    db2 = misc.derivative(lambda t: smoothed_double_beta(x, s1, b1, r1, s2, t, r2, constant, sigma=sigma), b2, dx=1e-3)
    dr2 = misc.derivative(lambda t: smoothed_double_beta(x, s1, b1, r1, s2, b2, t, constant, sigma=sigma), r2, dx=1e-3)
    dconstant = misc.derivative(lambda t: smoothed_double_beta(x, s1, b1, r1, s2, b2, r2, t, sigma=sigma), constant,
                                dx=1e-3)
    dsigma = misc.derivative(lambda t: smoothed_double_beta(x, s1, b1, r1, s2, b2, r2, constant, sigma=t), sigma,
                             dx=1e-3)
    return ds1, db1, dr1, ds2, db2, dr2, dconstant, dsigma


'''
class BrokenPowerLaw(Fittable1DModel):
    inputs = ("x",)
    outputs = ("y",)
    n = Parameter(default=1., min=1e-12, max=1e2, description="Normalisation factor at discontinuity")
    a1 = Parameter(default=1e-2, min=-3, max=10., description="Power law index one")
    a2 = Parameter(default=1., min=-3, max=10., description="Power law index two")
    r = Parameter(default=3., min=1e-1, max=1e2, description="Jump location")
    c = Parameter(default=2., min=1., max=10., description="Contraction factor")
    constant = Parameter(default=1e-5, min=1e-9, max=1, description="X-ray background")
    sigma = Parameter(default=0.05, min=0.05, max=1., fixed=True, description="Smooth kernel width")

    @staticmethod
    def evaluate(x, n, a1, a2, r, c, constant, sigma):
        if isinstance(x, np.ndarray):
            y = np.zeros(x.shape)
            for i in range(len(x)):
                y[i] = smoothed_broken_pwl(x[i], n, a1, a2, r, c, constant, sigma=sigma)
        elif isinstance(x, (int, float)):
            y = smoothed_broken_pwl(x, n, a1, a2, r, c, constant, sigma=sigma)
        return y

    @staticmethod
    def fit_deriv(x, n, a1, a2, r, c, constant, sigma):
        if isinstance(x, np.ndarray):
            dn = np.zeros_like(x)
            da1 = np.zeros_like(x)
            da2 = np.zeros_like(x)
            dr = np.zeros_like(x)
            dc = np.zeros_like(x)
            dconstant = np.zeros_like(x)
            dsigma = np.zeros_like(x)
            for i in range(len(x)):
                dn[i], da1[i], da2[i], dr[i], dc[i], dconstant[i], dsigma[i] = \
                    derive_smoothed_broken_pwl(x[i], n, a1, a2, r, c, constant, sigma=sigma)
        elif isinstance(x, (int, float)):
            dn, da1, da2, dr, dc, dconstant, dsigma = derive_smoothed_broken_pwl(x, n, a1, a2, r, c, constant,
                                                                                 sigma=sigma)
        return [dn, da1, da2, dr, dc, dconstant, dsigma]
'''


class Beta(Fittable1DModel):
    s = Parameter(default=1e-3, min=1e-10, max=1e2, description="Normalisation surface brightness")
    beta = Parameter(default=0.7, min=1e-10, max=5., description="Beta factor")
    r = Parameter(default=1.0, min=1e-1, max=20., description="Cut-off radius")
    constant = Parameter(default=1e-7, min=0, max=1e-3, description="X-ray background")
    sigma = Parameter(default=0.05, min=0.01, max=1., fixed=True, description="Smooth kernel width")

    @staticmethod
    def evaluate(x, s, beta, r, constant, sigma):
        if isinstance(x, np.ndarray):
            y = np.zeros(x.shape)
            for i in range(len(x)):
                y[i] = smoothed_beta(x[i], s, beta, r, constant, sigma=sigma)
        elif isinstance(x, (int, float)):
            y = smoothed_beta(x, s, beta, r, constant, sigma=sigma)
        return y

    @staticmethod
    def fit_deriv(x, s, beta, r, constant, sigma):
        if isinstance(x, np.ndarray):
            ds = np.zeros_like(x)
            dbeta = np.zeros_like(x)
            dr = np.zeros_like(x)
            dconstant = np.zeros_like(x)
            dsigma = np.zeros_like(x)
            for i in range(len(x)):
                ds[i], dbeta[i], dr[i], dconstant[i], dsigma[i] = \
                    derive_smoothed_beta(x[i], s, beta, r, constant, sigma=sigma)
        elif isinstance(x, (int, float)):
            ds, dbeta, dr, dconstant, dsigma = derive_smoothed_beta(x, s, beta, r, constant, sigma=sigma)
        return [ds, dbeta, dr, dconstant, dsigma]


class DoubleBeta(Fittable1DModel):
    s1 = Parameter(default=1e-3, min=1e-10, max=1e2, description="Normalisation surface brightness 1")
    beta1 = Parameter(default=0.7, min=1e-10, max=5., description="Beta factor 1")
    r1 = Parameter(default=1.0, min=1e-1, max=20., description="Cut-off radius 1")
    s2 = Parameter(default=1e-3, min=1e-10, max=1e2, description="Normalisation surface brightness 2")
    beta2 = Parameter(default=0.7, min=1e-10, max=5., description="Beta factor 2")
    r2 = Parameter(default=1.0, min=1e-1, max=20., description="Cut-off radius 2")
    constant = Parameter(default=1e-7, min=0, max=1e-3, description="X-ray background")
    sigma = Parameter(default=0.05, min=0.01, max=1., fixed=True, description="Smooth kernel width")

    @staticmethod
    def evaluate(x, s1, beta1, r1, s2, beta2, r2, constant, sigma):
        if isinstance(x, np.ndarray):
            y = np.zeros(x.shape)
            for i in range(len(x)):
                y[i] = smoothed_double_beta(x[i], s1, beta1, r1, s2, beta2, r2, constant, sigma=sigma)
        elif isinstance(x, (int, float)):
            y = smoothed_double_beta(x, s1, beta1, r1, s2, beta2, r2, constant, sigma=sigma)
        return y

    @staticmethod
    def fit_deriv(x, s1, beta1, r1, s2, beta2, r2, constant, sigma):
        if isinstance(x, np.ndarray):
            ds1 = np.zeros_like(x)
            dbeta1 = np.zeros_like(x)
            dr1 = np.zeros_like(x)
            ds2 = np.zeros_like(x)
            dbeta2 = np.zeros_like(x)
            dr2 = np.zeros_like(x)
            dconstant = np.zeros_like(x)
            dsigma = np.zeros_like(x)
            for i in range(len(x)):
                ds1[i], dbeta1[i], dr1[i], ds2[i], dbeta2[i], dr2[i], dconstant[i], dsigma[i] = \
                    derive_smoothed_double_beta(x[i], s1, beta1, r1, s2, beta2, r2, constant, sigma=sigma)
        elif isinstance(x, (int, float)):
            ds1, dbeta1, dr1, ds2, dbeta2, dr2, dconstant, dsigma = derive_smoothed_double_beta(x, s1, beta1, r1, s2,
                                                                                                beta2, r2, constant,
                                                                                                sigma=sigma)
        return ds1, dbeta1, dr1, ds2, dbeta2, dr2, dconstant, dsigma


def custom_model(func):
    """
    An alternative version of astropy.modeling.custom_model
    :param func:
    :return:
    """
    return _model_wrapper(func)


def _model_wrapper(func):
    name = func.__name__
    attr_dict = OrderedDict()

    # add parameters
    inputs, params = get_inputs_and_params(func)
    for para in params:
        attr_dict.update({para.name: Parameter(para.name, default=para.default)})
    # attr_dict.update({"sigma": Parameter("sigma", default=0.05, fixed=True)})

    attr_dict.update({"__module__": "model",
                      "__doc__": func.__doc__,
                      "n_inputs": len(inputs),
                      "n_outputs": len(inputs[0].name)})

    # evaluate method
    def evaluate(x, **evaluate_kwargs):

        if isinstance(x, np.ndarray):
            # y = np.zeros(x.shape)
            # for i in range(len(x)):
            #    y[i] = func(x[i], **evaluate_kwargs)
            # y = list(map(lambda t: func(t, **evaluate_kwargs), x))
            y = [func(x[i], **evaluate_kwargs) for i in range(len(x))]
            return np.array(y)
        elif isinstance(x, (int, float)):
            return func(x, **evaluate_kwargs)

    attr_dict.update({"evaluate": staticmethod(evaluate)})

    return type(name, (Model,), attr_dict)


@custom_model
def DoublePowerLaw(x, n=1, a1=0.1, a2=1.0, r=1.0, c=2.0, constant=1e-5):
    """
    x: radius for calculation;
    n: normalisation factor at the jump;
    a1: power law index 1;
    a2: power law index 2;
    r: jump location;
    c: contraction factor.

    x, r have a unit of arcmin;
    n has an arbitrary unit.
    """
    if x < r:
        # f = n ** 2 * (c ** 2 * integrate.quad(lambda z: ((x ** 2 + z ** 2) / r ** 2) ** -a1, 1 / np.inf,
        #                                      np.sqrt(r ** 2 - x ** 2), epsrel=1e-3)[0] +
        #              integrate.quad(lambda z: ((x ** 2 + z ** 2) / r ** 2) ** -a2, np.sqrt(r ** 2 - x ** 2),
        #                             np.inf, epsrel=1e-3)[0])
        f = n ** 2 * (c ** 2 * integrate.quad(_dpl_project, 1e-10, np.sqrt(r ** 2 - x ** 2),
                                              args=(x, r, a1), epsrel=1e-8)[0] +
                      integrate.quad(_dpl_project, np.sqrt(r ** 2 - x ** 2), np.inf, args=(x, r, a2),
                                     epsrel=1e-8)[0])
    else:
        # f = n ** 2 * integrate.quad(lambda z: ((x ** 2 + z ** 2) / r ** 2) ** -a2, 1e-4, np.inf, epsrel=1e-3)[0]
        f = n ** 2 * integrate.quad(_dpl_project, 1e-5, np.inf, args=(x, r, a2), epsrel=1e-10)[0]

    return f + constant


# divided functions for speed-up
@njit
def _dpl_project(z, x, r, a, ):
    return ((x ** 2 + z ** 2) / r ** 2) ** -a


@custom_model
def ConeDoublePowerLaw(theta, n=1, a1=0, a2=1.0, b=10., c=2.0, z1=0.5, z2=1.0, az=0.0, theta_max=70, constant=1e-5):
    def sb(n0, z, phi, b, c, a1, a2, az):
        n_in = lambda norm, rho, phi0, phi_b, alpha1, alphaz: norm * (phi0 / phi_b) ** (-alpha1) * rho ** (-alphaz)
        n_out = lambda norm, rho, phi0, phi_b, jump, alpha2, alphaz: \
            norm / jump * (phi0 / phi_b) ** (-alpha2) * rho ** (-alphaz)

        phi = np.abs(phi)
        out_bound = z * np.sqrt(1 - (np.cos(theta_max / 180 * np.pi) / np.cos(phi)) ** 2)
        n_in_sq = lambda l: n_in(n0, np.sqrt(z ** 2 + l ** 2),
                                 np.arccos(z * np.cos(phi) / np.sqrt(z ** 2 + l ** 2)), b, a1, az) ** 2
        n_out_sq = lambda l: n_out(n0, np.sqrt(z ** 2 + l ** 2),
                                   np.arctan(np.sqrt(z ** 2 * np.sin(phi) ** 2 + l ** 2) / (z * np.cos(phi))), b, c,
                                   a2, az) ** 2
        if phi > b:
            return 2 * integrate.quad(n_out_sq, 1e-7, out_bound)[0]
        else:
            l_b = z * np.sqrt(np.cos(phi) ** 2 / np.cos(b) ** 2 - 1)
            return 2 * (integrate.quad(n_in_sq, 1e-7, l_b)[0] + integrate.quad(n_out_sq, l_b, out_bound)[0])

    return integrate.quad(lambda t: t * sb(n, t, theta / 180 * np.pi, b / 180 * np.pi, c, a1, a2, az),
                          z1, z2)[0] * 2 / (z2 ** 2 - z1 ** 2) + constant
