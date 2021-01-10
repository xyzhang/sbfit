import numpy as np


def chi_square(y, y_model, err):
    return np.nansum((y - y_model) ** 2 / err ** 2)


def chi_square_deriv(y, y_model, err, model_deriv):
    chi2_deriv = np.zeros(len(model_deriv))
    for i in range(len(model_deriv)):
        chi2_deriv[i] = 2 * np.nansum((y_model - y) / err ** 2 * model_deriv[i])
    return chi2_deriv


def cstat(y, y_model):
    """
    Calculate c-stat value.

    Parameters
    ----------
    y : np.ndarray
        Observed values.
    y_model : np.ndarray
        Predicted values.
    Returns
    -------
    cstat : np.ndarray
        C-statistic value

    """
    cstat = 2 * np.nansum(y_model - y + y * np.log(y / y_model))
    return cstat


