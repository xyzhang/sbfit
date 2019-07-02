import numpy as np
from astropy.modeling.optimizers import Optimization
from astropy.modeling.fitting import Fitter
from .statistics import chi_square, chi_square_deriv, cstat, cstat_deriv


def lm_optimizer1D(x, y, err, theta, boundary, model, model_deriv, stat="ChiSquare", fudge=1e-4, acc=1e-3,
                   verbose=True):
    """
    LM_optimizer1D(x, y, err, theta, model, stat='ChiSquare', fudge=1e-4)

    Levenberg-Marquardt method for optimize model parameters for 1D data set.

    Parameters
    ----------
    x : array_like
        Input measurement points.
    y : array_like
        Measurement values.
    err : array_like
        Measurement 1 sigma uncertainties.
    theta : array_like
        Model initial parameters.
    boundary : 2*n array_like
        Upper and lower limit of parameters.
    model : callable
        Model function y = f(x, *theta).
    model_deriv : callable
        Model first order derivative function d_y/d_theta = f(x, *theta)
    stat : {'ChiSquare', 'C-Stat'}, optional
        Minimization statistical method.
    fudge : float, optional
        Fudge factor in Levenberg-Marquardt method.
    acc : float, optional
        Iteration accuracy.
    verbose : bool, optional
        Verbosity to show iteration information.
    """
    if stat == "ChiSquare":
        stat_func = chi_square
        stat_deriv_func = chi_square_deriv
    elif stat == "C-Stat":
        stat_func = cstat
        stat_deriv_func = cstat_deriv

    dstat = 100  # It is just an arbitrary initial big number
    stat_value = stat_func(y, model(x, *theta), err)
    print(stat_value)
    fail = 0
    # Start iteration
    mask_excess = np.zeros_like(theta) != 0
    while dstat > acc and fail <= 2:
        theta_deriv = np.array(model_deriv(x, *theta))
        # theta_deriv[mask_excess] = np.zeros_like(theta_deriv)[mask_excess]
        stat_deriv = stat_deriv_func(y, model(x, *theta), err, theta_deriv)
        if stat == "ChiSquare":
            theta_deriv_matrix = np.matrix(theta_deriv / err)  # should be different for different methods
        elif stat == "C-Stat":
            theta_deriv_matrix = np.matrix(theta_deriv * np.sqrt(y) / model(x, *theta))
        Hessian = theta_deriv_matrix * theta_deriv_matrix.T  # Hessian matrix
        Hessian_modified = Hessian + np.matrix(np.diag(np.diag(Hessian))) * fudge
        beta = np.matrix(-.5 * stat_deriv).T  # Be careful about the factor -0.5.
        dtheta = np.ravel(np.linalg.inv(Hessian_modified) * beta)
        new_theta = theta + dtheta
        # Check the boundary and update fitted value.
        # new_theta = (new_theta > boundary[0]) * boundary[0] + (new_theta <= boundary[0]) * new_theta
        # new_theta = (new_theta < boundary[1]) * boundary[1] + (new_theta >= boundary[1]) * new_theta
        maskup = new_theta > boundary[0]
        maskdown = new_theta < boundary[1]
        mask_excess = np.logical_or(maskup, maskdown)
        new_theta[maskup] = boundary[0][maskup]
        new_theta[maskdown] = boundary[1][maskdown]
        new_stat_value = stat_func(y, model(x, *(new_theta)), err)
        dstat = np.abs(stat_value - new_stat_value)
        if new_stat_value >= stat_value:
            fudge *= 10
            fail += 1
        else:
            fudge /= 10
            fail = 0
            stat_value = new_stat_value
            theta = new_theta
        if verbose:
            print(f"{stat} = {new_stat_value}\n{new_theta}")
    return theta, np.sqrt(np.linalg.inv(Hessian).diagonal()), stat_value


class LevenbergMarquardtOptimizer(Optimization):
    # TODO write a LM optimizer in astropy style.
    pass


# TODO write a simple LM fitter in astropy style.

class ChiSquareFitter(object):

    def __call__(self, model, x, y, yerr):
        pass


class CStatFitter(object):

    def __init__(self):
        self.acc = 1e-3

    def __call__(self, model, profile):
        # Get unfixed parameter values and boundaries.
        x = profile["r"]
        rawy = profile["cts"]
        exp = profile["exp"]
        bkg = profile["bkg_cts"]
        scale = profile["scale"]
        fixed_name = []
        fixed_value = []
        para_name = []
        para_init = []
        para_uplim = []
        para_lolim = []
        name: str
        for name in model.param_names:
            if not model.fixed[name]:
                para_name += [name]
                para_init += [model.__getattribute__(name).value]
                if model.bounds[name][1] == None:
                    para_uplim += [np.inf]
                else:
                    para_uplim += [model.bounds[name][1]]
                if model.bounds[name][0] == None:
                    para_lolim += [-np.inf]
                else:
                    para_lolim += [model.bounds[name][0]]
            else:
                fixed_name += [name]
                fixed_value += [model.__getattribute__(name).value]

        para_init = np.array(para_init)
        para_boundary = np.array([para_uplim, para_lolim])

        # Define evaluate function for optimizer.
        def f_evaluate(x, *theta, names=para_name, fixed=dict(zip(fixed_name, fixed_value))):
            return model.evaluate(x, **dict(zip(names, theta)), **fixed) * exp * scale + bkg

        def f_deriv(x, *theta, names=para_name, fixed=dict(zip(fixed_name, fixed_value))):
            deriv_all = model.fit_deriv(x, **dict(zip(names, theta)), **fixed)
            deriv = []
            for i in range(len(deriv_all)):
                if not list(model.fixed.values())[i]:
                    deriv += [deriv_all[i] * exp * scale]
            return deriv

        fit_para, fit_err, fit_stat = lm_optimizer1D(x, rawy, np.zeros_like(x), para_init, para_boundary, f_evaluate,
                                                     f_deriv, stat="C-Stat", acc=self.acc)
        for i in range(len(para_name)):
            model.__getattribute__(para_name[i]).value = fit_para[i]
        return model, fit_err, fit_stat
