import time
import copy
import itertools
import collections
import warnings

import numpy as np
from scipy import optimize, stats
from astropy import modeling
from astropy.table import Table
from astropy.convolution import CustomKernel, Gaussian1DKernel, convolve
import matplotlib.pyplot as plt
import emcee
import corner

from . import utils
from .statistics import cstat

__all__ = ["Profile"]


class Profile(object):
    def __init__(self, raw_data: Table, profile_axis="x", channel_width=1,
                 pixel_scale=1):
        self.profile_axis = profile_axis
        self._pixel_area = pixel_scale ** 2 * 3600  # arcmin^2 / pixel
        self.channel_width = channel_width
        self.channel_grid = np.arange(0, np.max(raw_data["r"]) + channel_width,
                                      channel_width)
        self.channel_center = self.channel_grid[:-1] + 0.5
        self.smooth_matrix = np.mat(np.diag(np.ones(len(self.channel_center))))
        self.smooth_fwhm = 0  # in _unit of grid
        self.raw_profile = self._create_channel(raw_data, self.channel_grid)
        self.bin_grid = None
        self.model = None
        self.mcmc_sampler = None
        self.error = collections.OrderedDict()
        self._show_stat = False
        self._min_stat = None
        self._error_approx = {}

    @staticmethod
    def _create_channel(raw_data, bin_grid):

        r = 0.5 * (bin_grid[:-1] + bin_grid[1:])
        cts_binned, _, _ = stats.binned_statistic(raw_data["r"],
                                                  raw_data["cts"],
                                                  statistic="sum",
                                                  bins=bin_grid)
        exp_binned, _, _ = stats.binned_statistic(raw_data["r"],
                                                  raw_data["exp"],
                                                  statistic="sum",
                                                  bins=bin_grid)
        raw_bkg_binned, _, _ = stats.binned_statistic(raw_data["r"],
                                                      raw_data["raw_bkg"],
                                                      statistic="sum",
                                                      bins=bin_grid)
        scaled_bkg_binned, _, _ = stats.binned_statistic(raw_data["r"],
                                                         raw_data[
                                                             "scaled_bkg"],
                                                         statistic="sum",
                                                         bins=bin_grid)

        return Table(
            [r, cts_binned, exp_binned, raw_bkg_binned, scaled_bkg_binned,
             np.zeros_like(r),
             -1 * np.ones_like(r)],
            names=("r", "cts", "exp", "raw_bkg", "scaled_bkg", "model_cts",
                   "bin_num"),
            dtype=(float, int, float, int, float, float, int))

    def deepcopy(self):
        return copy.deepcopy(self)

    def rebin(self, start=0, end=500, method="min_cts", min_cts=30,
              lin_width=10, log_width=0.05):
        """
        Rebin the profile from the raw profile.

        Parameters
        ----------
        start : float, optional
            Start point of the profile.
        end : float, optional
            End point of the profile. Depending on different methods, the actual boundary of the last
            bin can be different from this value.
        method : {"min_cts", "lin", "log"}, optional
            "min_cts" : each bin has at least a certain number of counts.
            "lin" : each bin has the same width.
            "log" : each bin has the same width in the logarithmic space.
        min_cts : int, optional
            The minimum count number in each of the bins. Used if method == "min_cts". 
        lin_width : float, optional
            The width of each bin. Used if method == "lin_width".
        log_width : float, optional
            The width of each bin in the logarithmic space, in _unit of dex. Used if method == "log_width".
        """

        # set the end of profile
        if end > np.max(self.raw_profile["r"]):
            bin_end = np.max(self.raw_profile["r"])
        else:
            bin_end = end

        # build bin grids for the specific method
        if method == "min_cts":
            if min_cts < 1:
                raise Exception("need at least one count per bin.")
            valid_range = self.raw_profile["r"] >= start
            bin_grid_loc_mask = np.diff(np.cumsum(
                self.raw_profile["cts"][valid_range]) // min_cts) >= 1

            bin_grid = np.append(np.array([start]),
                                 self.raw_profile["r"][valid_range][:-1][
                                     bin_grid_loc_mask])
            bin_grid = bin_grid[bin_grid <= end]
            self.bin_grid = bin_grid
        elif method == "lin":
            self.bin_grid = np.arange(start, bin_end + lin_width, lin_width)
        elif method == "log":
            if log_width <= 0:
                raise Exception("log_width must larger than 0")
            else:
                log_bin_grid = np.arange(np.log10(start),
                                         np.log10(bin_end) + log_width,
                                         log_width)
                bin_grid = 10 ** log_bin_grid
                pop_num = np.sum(np.diff(bin_grid) < 1)
                self.bin_grid = bin_grid[pop_num:]

        self._binning()

    def _binning(self):
        """
        A wrapper of the binning procedures.
        """
        bin_center = 0.5 * (self.bin_grid[:-1] + self.bin_grid[1:])

        cts_binned, _, bin_num = stats.binned_statistic(self.raw_profile["r"],
                                                        self.raw_profile[
                                                            "cts"],
                                                        statistic="sum",
                                                        bins=self.bin_grid)
        exp_binned, _, _ = stats.binned_statistic(self.raw_profile["r"],
                                                  self.raw_profile["exp"],
                                                  statistic="sum",
                                                  bins=self.bin_grid)
        raw_bkg_binned, _, _ = stats.binned_statistic(self.raw_profile["r"],
                                                      self.raw_profile[
                                                          "raw_bkg"],
                                                      statistic="sum",
                                                      bins=self.bin_grid)
        scaled_bkg_binned, _, _ = stats.binned_statistic(self.raw_profile["r"],
                                                         self.raw_profile[
                                                             "scaled_bkg"],
                                                         statistic="sum",
                                                         bins=self.bin_grid)
        model_cts_binned, _, _ = stats.binned_statistic(self.raw_profile["r"],
                                                        self.raw_profile[
                                                            "model_cts"],
                                                        statistic="sum",
                                                        bins=self.bin_grid)
        model_sb = model_cts_binned / exp_binned / self._pixel_area

        self.raw_profile["bin_num"] = bin_num

        norm_bkg_binned = scaled_bkg_binned / raw_bkg_binned
        norm_bkg_binned = np.nan_to_num(norm_bkg_binned, posinf=0, neginf=0)
        pseudo_bkgsb = scaled_bkg_binned / exp_binned / self._pixel_area
        pseudo_bkgsb_error = np.sqrt(
            raw_bkg_binned) * norm_bkg_binned / exp_binned / self._pixel_area
        net_cts_binned = cts_binned - scaled_bkg_binned
        sb_binned = net_cts_binned / exp_binned / \
                    self._pixel_area  # surface brightness
        sb_error = np.sqrt(
            cts_binned + raw_bkg_binned * norm_bkg_binned ** 2) / \
                   exp_binned / self._pixel_area
        r_error_left = bin_center - self.bin_grid[:-1]
        r_error_right = self.bin_grid[1:] - bin_center
        self.binned_profile = Table(
            [bin_center, r_error_left, r_error_right, sb_binned, sb_error,
             pseudo_bkgsb,
             pseudo_bkgsb_error, model_sb, cts_binned, scaled_bkg_binned],
            names=(
                "r", "r_error_left", "r_error_right", "_cone_sb", "sb_error",
                "bkg_sb",
                "bkg_sb_error", "model_sb", "total_cts", "bkg_cts"),
            dtype=(float, float, float, float, float, float, float, float, int,
                   float))

    def set_model(self, model: modeling.Model):
        """
        Set model for the Profile object.

        Parameters
        ----------
        model : Model
            The model used for fit.
        """
        self.model = model

    def calculate(self, update=True):
        """
        Calculate the C-statistic value for the current model.

        Parameters
        ----------
        update : bool, optional
            Update the model profile.

        Returns
        -------
        stat_value : float
            The calculated C-statistic value.
        """
        if self.model is None:
            Warning("No model found")
            return None
        else:
            tstart = time.time()
            self.model: modeling.Model
            valid_range_start = self.bin_grid[
                                    0] - self.smooth_fwhm * self.channel_width
            valid_range_end = self.bin_grid[
                                  -1] + self.smooth_fwhm * self.channel_width
            valid_grid_mask = np.logical_and(
                self.raw_profile["r"] <= valid_range_end,
                self.raw_profile["r"] >= valid_range_start)
            valid_channel_center = self.channel_center[valid_grid_mask]

            # calculate
            # model_value = self.model.evaluate(valid_channel_center,
            #                                  **dict(
            #                                      zip(self.model.param_names,
            #                                          self.model.parameters)))
            model_value = self.model.evaluate(valid_channel_center,
                                              *self.model.parameters)
            total_model_value = np.zeros_like(self.channel_center)
            total_model_value[valid_grid_mask] = model_value
            tmid = time.time()
            smoothed_model_value = self.smooth_matrix * np.mat(
                np.atleast_2d(total_model_value).T)
            smoothed_model_value = np.array(smoothed_model_value).flatten()
            smoothed_model_cts = smoothed_model_value * self.raw_profile[
                "exp"] * self._pixel_area
            self.raw_profile.replace_column("model_cts", smoothed_model_cts)

            model_cts_binned, _, _ = stats.binned_statistic(
                self.raw_profile["r"], self.raw_profile["model_cts"],
                statistic="sum", bins=self.bin_grid)
            if update:
                # update binned profile
                exp_binned, _, _ = stats.binned_statistic(
                    self.raw_profile["r"], self.raw_profile["exp"],
                    statistic="sum", bins=self.bin_grid)
                model_sb_binned = model_cts_binned / exp_binned / self._pixel_area
                self.binned_profile.replace_column("model_sb",
                                                   model_sb_binned, )

            # calculate residual
            model_total_cts = model_cts_binned + self.binned_profile["bkg_cts"]
            stat_value = cstat(self.binned_profile["total_cts"],
                               model_total_cts)
            tend = time.time()
            # print(tmid - tstart, tend - tstart)
            if self._show_stat:
                print(f"C-stat: {stat_value:.3f}")
            return stat_value

    def set_smooth_matrix(self, kernel_type, gaussian_sigma=1, lorenzian_x=1):
        if kernel_type == "identity":
            self.smooth_matrix = np.mat(
                np.diag(np.ones(len(self.channel_center))))
        if kernel_type == "gaussian":
            pass
        elif kernel_type == "lorenzian":
            pass

    def plot(self, plot_type="binned_profile", scale="loglog"):
        if plot_type == "binned_profile":
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.gca()
            ax.errorbar(self.binned_profile["r"],
                        self.binned_profile["_cone_sb"],
                        xerr=(self.binned_profile["r_error_right"],
                              self.binned_profile["r_error_left"]),
                        yerr=self.binned_profile["sb_error"], ls="",
                        label="_data", color="tab:blue")
            ax.errorbar(self.binned_profile["r"],
                        self.binned_profile["bkg_sb"],
                        xerr=(self.binned_profile["r_error_right"],
                              self.binned_profile["r_error_left"]),
                        yerr=self.binned_profile["bkg_sb_error"], ls="",
                        label="background", color="tab:green")
            ax.step(np.append(
                self.binned_profile["r"] - self.binned_profile["r_error_left"],
                np.array(self.binned_profile["r"][-1] +
                         self.binned_profile["r_error_right"][-1])),
                np.append(self.binned_profile["model_sb"],
                          np.array(self.binned_profile["model_sb"][-1])),
                where="post", label="model", color="tab:orange")
            if scale == "loglog":
                ax.loglog()
            elif scale == "semilogx":
                ax.semilogx()
            elif scale == "semilogy":
                ax.semilogy()
            elif scale == "linear":
                pass
            else:
                raise ValueError("Scale must be one of {'linear', 'loglog',"
                                 "'semilogx', 'semilogy'}.")
            ax.set_xlabel("r (arcsec)")
            ax.set_ylabel("SB")
            ax.legend()
            plt.show()
            plt.close(fig)
        elif plot_type == "mcmc_chain":
            if self.mcmc_sampler is None:
                warnings.warn("No sampler found, please run mcmc_error first.")
            else:
                samples: np.ndarray = self.mcmc_sampler.get_chain()
                fig, axes = plt.subplots(samples.shape[2], figsize=(10, 7),
                                         sharex="all")
                labels, _ = utils.get_free_parameter(self.model)
                for i in range(samples.shape[2]):
                    ax = axes[i]
                    ax.plot(samples[:, :, i], "k", alpha=0.3)
                    ax.set_xlim(0, len(samples))
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.1, 0.5)

                axes[-1].set_xlabel("step number")
                plt.show()
                plt.close(fig)

    def fit(self, show_step=False, show_result=True, record_fit=True,
            tolerance=1e-2):
        """
        Fit binned profile with the model.

        Parameters
        ----------
        show_step : bool, optional
            Whether show C-statistic value of each fit step.
        show_result : bool, optional
            Whether show fit result.
        record_fit : bool, optional
            Whether record the fit result in the Profile object.
        tolerance : float, optional
            Tolerance to terminate the fit iteration.

        Returns
        -------
        stat : float
            The best-fit C-statistic value.

        """
        if self.model is None:
            Warning("No model found")
        else:
            self.model: modeling.Model

            # fit
            stat, error = self._fit_wrapper(tol=tolerance, show_step=show_step)

            pnames_free, pvalues_free = utils.get_free_parameter(self.model)
            dof = len(self.binned_profile) - len(pnames_free)
            if show_result:
                print(f"Degree of freedom: {dof:d}; C-stat: {stat:.4f}")
                for item in pnames_free:
                    print(
                        f"{item}:\t{self.model.__getattribute__(item).value:.2e}")
                print(f"Uncertainties from rough estimation:")
                [print(f"{pnames_free[i]}: {error[i]:.3e}") for i in
                 range(len(pnames_free))]
            if record_fit:
                self._min_stat = stat
                self._error_approx = dict(zip(pnames_free, error))
            return stat

    def _fit_wrapper(self, tol=1e-2, show_step=True, nfail=10):
        """
        Fit routine using a Levenberg-Marquardt optimizer.

        Parameters
        ----------
        tol : float, optional
            Tolerance to terminate the fit iteration.
        show_step : float, optional
            Whether show C-statistic value of each fit step.
        nfail : int, optional
            The number of iteration before decreasing the statistics.

        Returns
        -------
        stat : float, optional
            The best-fit C-statistic value.
        errors : numpy.ndarray
            A rough estimation of the errors.
        """
        pnames_free, pvalues_free = utils.get_free_parameter(self.model)
        pvalues_free = np.array(pvalues_free)
        low_bounds, up_bounds = utils.get_parameter_bounds(self.model,
                                                           pnames_free)
        damp_factor = 1e-4
        dstat = np.inf
        stat = self.calculate(update=False)
        if show_step:
            print(f"Start fit\nC-stat: {stat:.3f}\n{pvalues_free}")
        fail = 0
        while dstat > tol and fail <= nfail:
            # calculate delta parameter
            alpha = 0.5 * self._stat_deriv_matrix()
            beta = -0.5 * np.mat(np.atleast_2d(
                self._stat_deriv()).T)  # Eq. 15.5.8 Numerical Recipes 3rd
            modified_alpha = alpha + np.diag(
                np.diag(alpha)) * damp_factor  # Eq. 15.5.13
            shift: np.matrix = np.linalg.inv(
                modified_alpha) * beta  # Eq. 15.5.14
            # shift = np.array(shift).flatten()
            new_pvalues_free = pvalues_free + np.array(shift).flatten()
            # check if new parameter vales reach the boundaries
            up_mask = new_pvalues_free > up_bounds
            new_pvalues_free[up_mask] = up_bounds[up_mask]
            low_mask = new_pvalues_free < low_bounds
            new_pvalues_free[low_mask] = low_bounds[low_mask]
            # set new parameter values
            for i in range(len(pnames_free)):
                self.model.__setattr__(pnames_free[i], new_pvalues_free[i])
            new_stat = self.calculate(update=False)
            if show_step:
                print(f"C-stat: {new_stat:.3f}\n{new_pvalues_free}")
            if stat - new_stat <= 0:  # new stat > current stat
                damp_factor *= 100
                fail += 1
                for i in range(len(pnames_free)):
                    self.model.__setattr__(pnames_free[i], pvalues_free[i])
            else:  # new stat < current stat
                dstat = stat - new_stat
                damp_factor /= 5
                fail = 0
                stat = new_stat
                _, pvalues_free = utils.get_free_parameter(self.model)
        stat = self.calculate(update=True)
        if show_step:
            print("Iteration terminated.")
        errors = np.array(np.sqrt(
            np.abs(np.linalg.inv(alpha).diagonal()))).flatten()
        return stat, errors

    def _stat_deriv(self, pnames=None):
        """
        Calculate the first derivative of the statistic value versus each parameter.

        Parameters
        ----------
        pnames : list, optional
            If set, only calculate the first derivative versus the given parameter.
        Returns
        -------
        deriv : numpy.ndarray
            The array of calculated derivatives.
        """
        pnames_free, pvalues_free = utils.get_free_parameter(self.model)
        current_model = copy.deepcopy(self.model)
        shift = 1e-4
        # calculate derivative for each parameter
        deriv = []
        pnames_deriv = []
        pvalues_deriv = []
        if pnames is not None:
            for name in pnames:
                if name not in pnames_free:
                    raise Exception(f"{name} is not a free parameter.")
                else:
                    index = pnames_free.index(name)
                    pnames_deriv += [pnames_free[index]]
                    pvalues_deriv += [pvalues_free[index]]
        else:
            pnames_deriv = pnames_free
            pvalues_deriv = pvalues_free

        for i in range(len(pnames_deriv)):
            self.model.__setattr__(pnames_deriv[i], pvalues_deriv[i])
            stat0 = self.calculate(update=False)
            self.model.__setattr__(pnames_deriv[i],
                                   1e-10 + pvalues_deriv[i] * (1 + shift))
            stat1 = self.calculate(update=False)
            deriv_value = (stat1 - stat0) / (1e-10 + pvalues_deriv[i] * shift)
            if not np.isnan(deriv_value):
                deriv += [deriv_value]
            else:
                deriv += [1e-5]
            # print(stat0, stat1, deriv)
            self.model = copy.deepcopy(current_model)
        deriv = np.array(deriv)
        return deriv

    def _stat_deriv_matrix(self):
        shift = 1e-4
        pnames_free, pvalues_free = utils.get_free_parameter(self.model)
        current_model = copy.deepcopy(self.model)
        deriv_matrix = np.zeros([len(pnames_free), len(pnames_free)])
        for comb in itertools.combinations(np.arange(len(pnames_free)), 2):
            stat0 = self._stat_deriv(pnames=[pnames_free[comb[0]]])[0]
            self.model.__setattr__(pnames_free[comb[1]],
                                   pvalues_free[comb[1]] * (1 + shift) + 1e-10)
            stat1 = self._stat_deriv(pnames=[pnames_free[comb[0]]])[0]
            self.model = copy.deepcopy(current_model)
            second_deriv = (stat1 - stat0) / (
                    1e-10 + pvalues_free[comb[1]] * shift)
            deriv_matrix[comb] = second_deriv
        deriv_matrix += deriv_matrix.T
        for i in range(len(pnames_free)):
            stat0 = self._stat_deriv(pnames=[pnames_free[i]])[0]
            self.model.__setattr__(pnames_free[i],
                                   pvalues_free[i] * (1 + shift) + 1e-10)
            stat1 = self._stat_deriv(pnames=[pnames_free[i]])[0]
            self.model = copy.deepcopy(current_model)
            second_deriv = (stat1 - stat0) / (1e-10 + pvalues_free[i] * shift)
            deriv_matrix[i, i] = second_deriv
        return np.mat(deriv_matrix)

    def error_estimate_delta_stat(self, param: str, sigma=1, show_step=True):

        self.model: modeling.Model
        if self._min_stat is None:  # check if self._min_stat exists
            warnings.warn("Fit the profile first.")
        elif param not in self.model.param_names:  # check if the model has the parameter
            warnings.warn(f"Parameter '{param}' is not in the model.")
        elif self.model.fixed[param]:
            warnings.warn(f"Parameter '{param}' is fixed.")
        else:
            param_value = self.model.__getattribute__(param).value

            best_model = copy.deepcopy(
                self.model)  # a backup of the best-fit model

            # solve the function delta-cstat = 1
            self.model.fixed[param] = True
            self.model.bounds[param] = (
                param_value,
                best_model.bounds[param][1])  # constraint lower limit
            init_error = self._error_approx[param] * np.array([1, -1])
            error = []
            for i in range(2):
                root = optimize.root_scalar(self._root_solve_func,
                                            x0=param_value + init_error[
                                                i] * sigma,
                                            args=(
                                                param, sigma ** 2, show_step),
                                            method="newton",
                                            fprime=self._root_solve_derivative,
                                            xtol=1e-2)
                print(root)
                error[i] = root[0]

            self.model = best_model

    def _root_solve_func(self, x, param, dstat, show_step):

        self.model.__setattr__(param, x)
        stat = self.fit(show_step=False, show_result=False, record_fit=False,
                        tolerance=1e-2)
        if show_step:
            print(
                f"{param} = {x:.2f}; delta C-stat = {stat - self._min_stat:.3e}")
        return stat - self._min_stat - dstat

    def _root_solve_derivative(self, x, param, dstat, show_step):
        self.model.__setattr__(param, x * (1 + 1e-4))
        stat1 = self.fit(show_step=False, show_result=False, record_fit=False)
        self.model.__setattr__(param, x)
        stat0 = self.fit(show_step=False, show_result=False, record_fit=False)
        return (stat1 - stat0) / (x * 1e-4)

    def mcmc_error(self, nsteps=5000, nwalkers=32, burnin=500):
        pnames_free, pvalues_free = utils.get_free_parameter(self.model)
        model = copy.deepcopy(self.model)  # a backup
        pos = np.array(pvalues_free) + 1e-4 * np.random.randn(nwalkers, len(
            pvalues_free)) * np.array(pvalues_free)
        sampler = emcee.EnsembleSampler(nwalkers, len(pvalues_free),
                                        self._mcmc_log_probability, )
        sampler.run_mcmc(pos, nsteps, progress=True)

        # store smapler
        self.mcmc_sampler = sampler

        # output error
        flat_samples = sampler.get_chain(discard=burnin, flat=True)
        self.error = collections.OrderedDict()
        for i in range(pnames_free):
            left, mid, right = np.percentile(flat_samples[:, i], [16, 50, 84])
            self.error.update({pnames_free[i]: (right - mid, mid - left)})
            self.model.__setattr__(pnames_free[i], mid)

    def _mcmc_log_probability(self, theta):
        pnames_free, _ = utils.get_free_parameter(self.model)
        low_bounds, up_bounds = utils.get_parameter_bounds(self.model,
                                                           pnames_free)
        if np.sum(np.array(theta) <= up_bounds) + np.sum(
                np.array(theta) >= low_bounds) == 2 * len(pnames_free):
            for i in range(len(pnames_free)):
                self.model.__setattr__(pnames_free[i], theta[i])
            return -0.5 * self.calculate(update=False)
        else:
            return -np.inf
