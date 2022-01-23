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
    mask_0 = y <= 0
    mask_no0 = np.logical_not(mask_0)

    comp1 = 2 * np.sum((y_model - y)[mask_0])
    comp2 = 2 * np.sum(y_model[mask_no0] - y[mask_no0] +
                       y[mask_no0] * np.log(y[mask_no0] / y_model[mask_no0]))
    cstat = comp1 + comp2
    return cstat
