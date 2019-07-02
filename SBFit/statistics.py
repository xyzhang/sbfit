import numpy as np


def chi_square(y, y_model, err):
    return np.nansum((y - y_model) ** 2 / err ** 2)


def chi_square_deriv(y, y_model, err, model_deriv):
    chi2_deriv = np.zeros(len(model_deriv))
    for i in range(len(model_deriv)):
        chi2_deriv[i] = 2 * np.nansum((y_model - y) / err ** 2 * model_deriv[i])
    return chi2_deriv


def cstat(y, y_model, err):
    return 2 * np.nansum(y_model - y + y * np.log(y / y_model))


def cstat_deriv(y, y_model, err, model_deriv):
    c_deriv = np.zeros(len(model_deriv))
    for i in range(len(model_deriv)):
        c_deriv[i] = 2 * np.nansum(model_deriv[i] * (1 - y / y_model))
    return c_deriv
