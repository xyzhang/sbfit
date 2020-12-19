import time
import copy
import itertools
import collections
import warnings
import numpy as np
from scipy import optimize, stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import table, modeling
from astropy.table import Table, vstack, hstack, Column
from astropy.convolution import CustomKernel, Gaussian1DKernel, convolve
import matplotlib.pyplot as plt
import emcee
import corner
from .region import Epanda, Panda, Circle, Projection, Ellipse
from .exception import *
from . import utils
from .utils import xy2elliptic, xyrot, isincircle, isinellipse
from .image import CtsImageList, ExpImageList, BkgImageList
from .statistics import cstat


class Profile(object):
    def __init__(self, raw_data: Table, ptype="radial", channel_width=1, xscale=1, yscale=1):
        self.ptype = ptype
        self.pixel_area = xscale * yscale * 3600  # arcmin^2 / pixel
        self.channel_width = channel_width
        self.channel_grid = np.arange(0, np.max(raw_data["r"]) + channel_width, channel_width)
        self.channel_center = self.channel_grid[:-1] + 0.5
        self.smooth_matrix = np.mat(np.diag(np.ones(len(self.channel_center))))
        self.smooth_fwhm = 0  # in unit of grid
        self.raw_profile = self.__create_channel(raw_data, self.channel_grid)
        self.bin_grid = None
        self.model = None
        self.mcmc_sampler = None
        self.error = collections.OrderedDict()
        self.__show_stat = False
        self.__min_stat = None
        self.__error_approx = {}

    @staticmethod
    def __create_channel(raw_data, bin_grid):

        r = 0.5 * (bin_grid[:-1] + bin_grid[1:])
        cts_binned, _, _ = stats.binned_statistic(raw_data["r"], raw_data["cts"], statistic="sum", bins=bin_grid)
        exp_binned, _, _ = stats.binned_statistic(raw_data["r"], raw_data["exp"], statistic="sum", bins=bin_grid)
        raw_bkg_binned, _, _ = stats.binned_statistic(raw_data["r"], raw_data["raw_bkg"], statistic="sum",
                                                      bins=bin_grid)
        scaled_bkg_binned, _, _ = stats.binned_statistic(raw_data["r"], raw_data["scaled_bkg"], statistic="sum",
                                                         bins=bin_grid)

        return Table([r, cts_binned, exp_binned, raw_bkg_binned, scaled_bkg_binned, np.zeros_like(r),
                      -1 * np.ones_like(r)],
                     names=("r", "cts", "exp", "raw_bkg", "scaled_bkg", "model_cts", "bin_num"),
                     dtype=(float, int, float, int, float, float, int))

    def rebin(self, start=0, end=500, method="min_cts", min_cts=30, lin_width=10, log_width=0.05):
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
            The width of each bin in the logarithmic space, in unit of dex. Used if method == "log_width".
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
            bin_grid_loc_mask = np.diff(np.cumsum(self.raw_profile["cts"][valid_range]) // min_cts) >= 1

            bin_grid = np.append(np.array([start]), self.raw_profile["r"][valid_range][:-1][bin_grid_loc_mask])
            bin_grid = bin_grid[bin_grid <= end]
            self.bin_grid = bin_grid
        elif method == "lin":
            self.bin_grid = np.arange(start, bin_end + lin_width, lin_width)
        elif method == "log":
            if log_width <= 0:
                raise Exception("log_width must larger than 0")
            else:
                log_bin_grid = np.arange(np.log10(start), np.log10(bin_end) + log_width, log_width)
                bin_grid = 10 ** log_bin_grid
                pop_num = np.sum(np.diff(bin_grid) < 1)
                self.bin_grid = bin_grid[pop_num:]

        self.__binning()

    def __binning(self):
        """
        A wrapper of the binning procedures.
        """
        bin_center = 0.5 * (self.bin_grid[:-1] + self.bin_grid[1:])

        cts_binned, _, bin_num = stats.binned_statistic(self.raw_profile["r"], self.raw_profile["cts"],
                                                        statistic="sum", bins=self.bin_grid)
        exp_binned, _, _ = stats.binned_statistic(self.raw_profile["r"], self.raw_profile["exp"],
                                                  statistic="sum", bins=self.bin_grid)
        raw_bkg_binned, _, _ = stats.binned_statistic(self.raw_profile["r"], self.raw_profile["raw_bkg"],
                                                      statistic="sum", bins=self.bin_grid)
        scaled_bkg_binned, _, _ = stats.binned_statistic(self.raw_profile["r"], self.raw_profile["scaled_bkg"],
                                                         statistic="sum", bins=self.bin_grid)
        model_cts_binned, _, _ = stats.binned_statistic(self.raw_profile["r"], self.raw_profile["model_cts"],
                                                        statistic="sum", bins=self.bin_grid)
        model_sb = model_cts_binned / exp_binned / self.pixel_area

        self.raw_profile["bin_num"] = bin_num

        norm_bkg_binned = scaled_bkg_binned / raw_bkg_binned
        norm_bkg_binned = np.nan_to_num(norm_bkg_binned, posinf=0, neginf=0)
        pseudo_bkgsb = scaled_bkg_binned / exp_binned / self.pixel_area
        pseudo_bkgsb_error = np.sqrt(raw_bkg_binned) * norm_bkg_binned / exp_binned / self.pixel_area
        net_cts_binned = cts_binned - scaled_bkg_binned
        sb_binned = net_cts_binned / exp_binned / self.pixel_area  # surface brightness
        sb_error = np.sqrt(cts_binned + raw_bkg_binned * norm_bkg_binned ** 2) / exp_binned / self.pixel_area
        r_error_left = bin_center - self.bin_grid[:-1]
        r_error_right = self.bin_grid[1:] - bin_center
        self.binned_profile = Table([bin_center, r_error_left, r_error_right, sb_binned, sb_error, pseudo_bkgsb,
                                     pseudo_bkgsb_error, model_sb, cts_binned, scaled_bkg_binned],
                                    names=("r", "r_error_left", "r_error_right", "sb", "sb_error", "bkg_sb",
                                           "bkg_sb_error", "model_sb", "total_cts", "bkg_cts"),
                                    dtype=(float, float, float, float, float, float, float, float, int, float))

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
            valid_range_start = self.bin_grid[0] - self.smooth_fwhm * self.channel_width
            valid_range_end = self.bin_grid[-1] + self.smooth_fwhm * self.channel_width
            valid_grid_mask = np.logical_and(self.raw_profile["r"] <= valid_range_end,
                                             self.raw_profile["r"] >= valid_range_start)
            valid_channel_center = self.channel_center[valid_grid_mask]

            # calculate
            model_value = self.model.evaluate(valid_channel_center,
                                              **dict(zip(self.model.param_names, self.model.parameters)))
            total_model_value = np.zeros_like(self.channel_center)
            total_model_value[valid_grid_mask] = model_value
            tmid = time.time()
            smoothed_model_value = self.smooth_matrix * np.mat(np.atleast_2d(total_model_value).T)
            smoothed_model_value = np.array(smoothed_model_value).flatten()
            smoothed_model_cts = smoothed_model_value * self.raw_profile["exp"] * self.pixel_area
            self.raw_profile.replace_column("model_cts", smoothed_model_cts)

            model_cts_binned, _, _ = stats.binned_statistic(self.raw_profile["r"], self.raw_profile["model_cts"],
                                                            statistic="sum", bins=self.bin_grid)
            if update:
                # update binned profile
                exp_binned, _, _ = stats.binned_statistic(self.raw_profile["r"], self.raw_profile["exp"],
                                                          statistic="sum", bins=self.bin_grid)
                model_sb_binned = model_cts_binned / exp_binned / self.pixel_area
                self.binned_profile.replace_column("model_sb", model_sb_binned, )

            # calculate residual
            model_total_cts = model_cts_binned + self.binned_profile["bkg_cts"]
            stat_value = cstat(self.binned_profile["total_cts"], model_total_cts)
            tend = time.time()
            # print(tmid - tstart, tend - tstart)
            if self.__show_stat:
                print(f"C-stat: {stat_value:.3f}")
            return stat_value

    def set_smooth_matrix(self, kernel_type, gaussian_sigma=1, lorenzian_x=1):
        if kernel_type == "identity":
            self.smooth_matrix = np.mat(np.diag(np.ones(len(self.channel_center))))
        if kernel_type == "gaussian":
            pass
        elif kernel_type == "lorenzian":
            pass

    def plot(self, plot_type="binned_profile"):
        if plot_type == "binned_profile":
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.gca()
            ax.errorbar(self.binned_profile["r"], self.binned_profile["sb"],
                        xerr=(self.binned_profile["r_error_right"], self.binned_profile["r_error_left"]),
                        yerr=self.binned_profile["sb_error"], ls="", label="data", color="tab:blue")
            ax.errorbar(self.binned_profile["r"], self.binned_profile["bkg_sb"],
                        xerr=(self.binned_profile["r_error_right"], self.binned_profile["r_error_left"]),
                        yerr=self.binned_profile["bkg_sb_error"], ls="", label="background", color="tab:green")
            ax.step(np.append(self.binned_profile["r"] - self.binned_profile["r_error_left"],
                              np.array(self.binned_profile["r"][-1] + self.binned_profile["r_error_right"][-1])),
                    np.append(self.binned_profile["model_sb"], np.array(self.binned_profile["model_sb"][-1])),
                    where="post", label="model", color="tab:orange")
            ax.loglog()
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
                fig, axes = plt.subplots(samples.shape[2], figsize=(10, 7), sharex="all")
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

    def fit(self, show_step=False, show_result=True, record_fit=True, tolerance=1e-2):
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
            stat, error = self.__fit_wrapper(tol=tolerance, show_step=show_step)

            pnames_free, pvalues_free = utils.get_free_parameter(self.model)
            dof = len(self.binned_profile) - len(pnames_free)
            if show_result:
                print(f"Degree of freedom: {dof:d}; C-stat: {stat:.4f}")
                for item in pnames_free:
                    print(f"{item}:\t{self.model.__getattribute__(item).value:.2e}")
                print(f"Uncertainties from rough estimation:")
                [print(f"{pnames_free[i]}: {error[i]:.3e}") for i in range(len(pnames_free))]
            if record_fit:
                self.__min_stat = stat
                self.__error_approx = dict(zip(pnames_free, error))
            return stat

    def __fit_wrapper(self, tol=1e-2, show_step=True, nfail=10):
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
        low_bounds, up_bounds = utils.get_parameter_bounds(self.model, pnames_free)
        damp_factor = 1e-4
        dstat = np.inf
        stat = self.calculate(update=False)
        if show_step:
            print(f"Start fit\nC-stat: {stat:.3f}")
        fail = 0
        while dstat > tol and fail <= nfail:
            # calculate delta parameter
            alpha = 0.5 * self.__stat_deriv_matrix()
            beta = -0.5 * np.mat(np.atleast_2d(self.__stat_deriv()).T)  # Eq. 15.5.8 Numerical Recipes 3rd
            modified_alpha = alpha + np.diag(np.diag(alpha)) * damp_factor  # Eq. 15.5.13
            shift: np.matrix = np.linalg.inv(modified_alpha) * beta  # Eq. 15.5.14
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
                print(f"C-stat: {new_stat:.3f}")
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
        errors = np.array(np.sqrt(np.linalg.inv(alpha).diagonal())).flatten(),
        return stat, errors

    def __stat_deriv(self, pnames=None):
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
        shift = 1e-2
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
            self.model.__setattr__(pnames_deriv[i], pvalues_deriv[i] * (1 + shift))
            stat1 = self.calculate(update=False)
            deriv_value = (stat1 - stat0) / pvalues_deriv[i] / shift
            if not np.isnan(deriv_value):
                deriv += [deriv_value]
            else:
                deriv += [1e-5]
            # print(stat0, stat1, deriv)
            self.model = copy.deepcopy(current_model)
        deriv = np.array(deriv)
        return deriv

    def __stat_deriv_matrix(self):
        pnames_free, pvalues_free = utils.get_free_parameter(self.model)
        current_model = copy.deepcopy(self.model)
        deriv_matrix = np.zeros([len(pnames_free), len(pnames_free)])
        for comb in itertools.combinations(np.arange(len(pnames_free)), 2):
            stat0 = self.__stat_deriv(pnames=[pnames_free[comb[0]]])[0]
            self.model.__setattr__(pnames_free[comb[1]], pvalues_free[comb[1]] * (1 + 1e-4))
            stat1 = self.__stat_deriv(pnames=[pnames_free[comb[0]]])[0]
            self.model = copy.deepcopy(current_model)
            second_deriv = (stat1 - stat0) / pvalues_free[comb[1]] / 1e-4
            deriv_matrix[comb] = second_deriv
        deriv_matrix += deriv_matrix.T
        for i in range(len(pnames_free)):
            stat0 = self.__stat_deriv(pnames=[pnames_free[i]])[0]
            self.model.__setattr__(pnames_free[i], pvalues_free[i] * (1 + 1e-4))
            stat1 = self.__stat_deriv(pnames=[pnames_free[i]])[0]
            self.model = copy.deepcopy(current_model)
            second_deriv = (stat1 - stat0) / pvalues_free[i] / 1e-4
            deriv_matrix[i, i] = second_deriv
        return np.mat(deriv_matrix)

    def error_estimate_delta_stat(self, param: str, sigma=1, show_step=True):

        self.model: modeling.Model
        if self.__min_stat is None:  # check if self.__min_stat exists
            warnings.warn("Fit the profile first.")
        elif param not in self.model.param_names:  # check if the model has the parameter
            warnings.warn(f"Parameter '{param}' is not in the model.")
        elif self.model.fixed[param]:
            warnings.warn(f"Parameter '{param}' is fixed.")
        else:
            param_value = self.model.__getattribute__(param).value

            best_model = copy.deepcopy(self.model)  # a backup of the best-fit model

            # solve the function delta-cstat = 1
            self.model.fixed[param] = True
            self.model.bounds[param] = (param_value, best_model.bounds[param][1])  # constraint lower limit
            init_error = self.__error_approx[param] * np.array([1, -1])
            error = []
            for i in range(2):
                root = optimize.root_scalar(self.__root_solve_func, x0=param_value + init_error[i] * sigma,
                                            args=(param, sigma ** 2, show_step),
                                            method="newton", fprime=self.__root_solve_derivative, xtol=1e-2)
                print(root)
                error[i] = root[0]

            self.model = best_model

    def __root_solve_func(self, x, param, dstat, show_step):

        self.model.__setattr__(param, x)
        stat = self.fit(show_step=False, show_result=False, record_fit=False, tolerance=1e-2)
        if show_step:
            print(f"{param} = {x:.2f}; delta C-stat = {stat - self.__min_stat:.3e}")
        return stat - self.__min_stat - dstat

    def __root_solve_derivative(self, x, param, dstat, show_step):
        self.model.__setattr__(param, x * (1 + 1e-4))
        stat1 = self.fit(show_step=False, show_result=False, record_fit=False)
        self.model.__setattr__(param, x)
        stat0 = self.fit(show_step=False, show_result=False, record_fit=False)
        return (stat1 - stat0) / (x * 1e-4)

    def mcmc_error(self, nsteps=5000, nwalkers=32, burnin=500):
        pnames_free, pvalues_free = utils.get_free_parameter(self.model)
        model = copy.deepcopy(self.model)  # a backup
        pos = np.array(pvalues_free) + 1e-4 * np.random.randn(nwalkers, len(pvalues_free)) * np.array(pvalues_free)
        sampler = emcee.EnsembleSampler(nwalkers, len(pvalues_free), self.__log_probability, )
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

    def __log_probability(self, theta):
        pnames_free, _ = utils.get_free_parameter(self.model)
        low_bounds, up_bounds = utils.get_parameter_bounds(self.model, pnames_free)
        # lp = self.__log_prior(theta)
        # if np.isinf(lp):
        if np.sum(np.array(theta) <= up_bounds) + np.sum(np.array(theta) >= low_bounds) == 2 * len(pnames_free):
            for i in range(len(pnames_free)):
                self.model.__setattr__(pnames_free[i], theta[i])
            return -0.5 * self.calculate(update=False)
        else:
            return -np.inf


class DataSet(object):
    data: Table
    ctsimages: CtsImageList
    expimages: ExpImageList
    bkgimages: BkgImageList

    def __init__(self, ctsimagelist, expimagelist, bkgimagelist, region_list):
        self.data = Table()
        nctsimage = len(ctsimagelist.data)
        nexpimage = len(expimagelist.data)
        nbkgimage = len(bkgimagelist.data)
        # Initialisation checking.
        # Check numbers of images.
        if nctsimage != nexpimage:
            raise MismatchingError("Number of exposure maps is not equal to number of count maps.")
        if nctsimage != nbkgimage:
            raise MismatchingError("Number of background maps is not equal to number of count maps.")
        self.ctsimages = ctsimagelist
        self.expimages = expimagelist
        self.bkgimages = bkgimagelist
        self.xscale = 1.
        self.yscale = 1.
        # Check region numbers.
        self.add_region = region_list.add
        self.sub_region = region_list.sub
        # Check region frame.
        if self.add_region.frame not in ["fk4", "fk5", "icrs", "galactic", "ecliptic"]:
            warnings.warn("The coordinate frame is not celestial.")
        if not (isinstance(self.add_region, Panda) or isinstance(self.add_region, Epanda) or isinstance(self.add_region,
                                                                                                        Projection)):
            raise TypeError("Add region must be Panda or Epanda")
        self.get_data()

    def get_data(self):
        """Collecting data from images"""
        add_region = self.add_region
        # check additive region type
        if isinstance(add_region, Epanda) or isinstance(add_region, Panda):
            startangle = add_region.startangle
            stopangle = add_region.stopangle
            startangle += add_region.angle
            stopangle += add_region.angle
            if startangle > 360:
                startangle -= 360
                stopangle -= 360

        # Image size check.
        for i in range(len(self.ctsimages.data)):
            if self.ctsimages.data[i].shape != self.expimages.data[i].shape or \
                    self.ctsimages.data[i].shape != self.bkgimages.data[i].shape:
                raise MismatchingError("Count map, exposure map and background map should have the same shape.")
        # TODO Add SUB region features.

        # For each observation.
        sub_data = []
        for i in range(len(self.ctsimages.data)):
            wcs: WCS = self.ctsimages.wcses[i]
            ly, lx = self.ctsimages.data[i].shape
            xcoor, ycoor = np.meshgrid(np.arange(lx) + 1., np.arange(ly) + 1.)
            subtraction_mask = np.zeros_like(xcoor)
            if add_region.frame in ["fk4", "fk5", "icrs", "galactic", "ecliptic"]:
                xscale, yscale = proj_plane_pixel_scales(wcs)
                self.xscale = xscale
                self.yscale = yscale
                # Region coordinates transfer.
                if isinstance(add_region, Panda) or isinstance(add_region, Epanda):
                    x0, y0 = wcs.all_world2pix(np.array([[add_region.x.value, add_region.y.value]]), 1)[0]
                elif isinstance(add_region, Projection):
                    width = add_region.width.value / 3600 / xscale  # Unit: pixel
                    xstart, ystart = \
                        wcs.all_world2pix(np.array([[add_region.xstart.value, add_region.ystart.value]]), 1)[0]
                    xend, yend = wcs.all_world2pix(np.array([[add_region.xend.value, add_region.yend.value]]), 1)[0]

                # FITS image coordinate order should start from 1.
                # Build grid of pixels.
                if isinstance(add_region, Panda) or isinstance(add_region, Epanda):
                    outermajor = add_region.outermajor.value / 3600 / xscale  # Unit: pixel
                    outerminor = add_region.outerminor.value / 3600 / xscale
                    innermajor = add_region.innermajor.value / 3600 / xscale
                    innerminor = add_region.innerminor.value / 3600 / xscale
                # mask point sources
                for sub_region in self.sub_region:
                    if isinstance(sub_region, Circle):
                        xsub, ysub = wcs.all_world2pix(np.array([[sub_region.x.value, sub_region.y.value]]), 1)[0]
                        rsub = sub_region.radius.value / 3600 / xscale
                        subtraction_mask = np.logical_or(isincircle(xcoor, ycoor, xsub, ysub, rsub), subtraction_mask)
                    elif isinstance(sub_region, Ellipse):
                        xsub, ysub = wcs.all_world2pix(np.array([[sub_region.x.value, sub_region.y.value]]), 1)[0]
                        majorsub = sub_region.major.value / 3600 / xscale
                        minorsub = sub_region.major.value / 3600 / xscale
                        pasub = sub_region.pa
                        subtraction_mask = np.logical_or(
                            isinellipse(xcoor, ycoor, xsub, ysub, majorsub, minorsub, pasub), subtraction_mask)
                    else:
                        warnings.warn("So far only Circle subtraction region is supported.")
            else:
                # TODO Add features for non-celestial regions.
                pass
            exist_mask = (self.expimages.data[i] * 1.) > 0
            exist_mask = np.logical_and(exist_mask, np.logical_not(subtraction_mask))
            if isinstance(add_region, Panda) or isinstance(add_region, Epanda):
                az, r_pix = xy2elliptic(xcoor, ycoor, x0, y0, outermajor, outerminor, add_region.angle, startangle,
                                        stopangle)
                # rad_mask = np.logical_and(r_pix >= innermajor, r_pix < outermajor)
                # rad_mask = r_pix >= innermajor
                if add_region.axis == "r":
                    if stopangle > 360:
                        az_mask = np.logical_or(az >= startangle, az < stopangle - 360)
                    else:
                        az_mask = np.logical_and(az >= startangle, az < stopangle)
                    # valid_mask = np.logical_and(exist_mask, np.logical_and(rad_mask, az_mask))
                    valid_mask = np.logical_and(exist_mask, az_mask)
                    r_pix_valid = r_pix[valid_mask]
                    r_arcmin_valid = r_pix_valid * xscale * 60
                    # az_valid = az[valid_mask]
                    # lx_valid = lx[valid_mask]
                    # ly_valid = ly[valid_mask]
                elif add_region.axis == "theta":
                    rad_mask = np.logical_and(r_pix >= innermajor, r_pix < outermajor)
                    az -= startangle
                    valid_mask = np.logical_and(exist_mask, rad_mask)
                    r_arcmin_valid = az[valid_mask]

            elif isinstance(add_region, Projection):
                r_pix, ry = xyrot(xcoor, ycoor, xstart, ystart, xend, yend)
                valid_mask = np.logical_and(exist_mask,
                                            np.logical_and(r_pix >= 0, np.logical_and(ry >= 0, ry <= width)))
                r_pix_valid = r_pix[valid_mask]
                r_arcmin_valid = r_pix_valid * xscale * 60

            # valid pixels
            cts_valid = self.ctsimages.data[i][valid_mask]
            exp_valid = self.expimages.data[i][valid_mask]
            bkg_valid = self.bkgimages.data[i][valid_mask]
            # For different scaling method.
            if self.bkgimages.norm_type == "count":
                bkgnorm = self.bkgimages.bkgnorm[i]
            elif self.bkgimages.norm_type == "flux":
                bkgnorm = self.bkgimages.bkgnorm[i] * self.ctsimages.exptime[i] / self.bkgimages.exptime[i]
            net_cts_valid = cts_valid - bkg_valid * bkgnorm
            sub_data += [
                Table(
                    np.array([r_arcmin_valid * 60, cts_valid, exp_valid, bkg_valid, bkg_valid * bkgnorm, net_cts_valid,
                              bkgnorm * np.ones_like(cts_valid), i * np.ones(len(cts_valid))]).T,
                    names=("r", "cts", "exp", "raw_bkg", "scaled_bkg", "net_cts", "bkgnorm", "i"),
                    dtype=(float, float, float, float, float, float, float, int))]
            self.data = vstack(sub_data)
            self.data.sort("r")
            return self.data, self.xscale

    def get_profile(self, rmin, rmax, channelsize=1, bin_method="custom", min_cts=50):
        """
        To get a surface brightness profile.
        astropy.table.Table is used for grouping radius bins.
        rmin, rmax have a unit of arcmin.
        channelsize has a unit of arcsec
        """
        # Purify data set.
        data = self.data
        data = data[data["r"] >= rmin]
        data = data[data["r"] < rmax]
        add_region = self.add_region

        if bin_method == "custom":
            # Define default channels.
            # channels_min = np.floor(add_region.innermajor * 60)
            # channels_max = np.ceil(add_region.outermajor * 60)
            channels_min = np.floor(rmin * 60)
            channels_max = np.ceil(rmax * 60)
            channels = np.arange(channels_min, channels_max, channelsize) / 60  # Unit: arcmin
            if channels[-1] < rmax:
                channels = np.append(channels, [rmax])
        elif bin_method == "dynamic":
            cts_cum = np.cumsum(data["net_cts"])
            channels = np.zeros(int(np.max(cts_cum / min_cts)))
            for i in range(len(channels)):
                channels[i] = data["r"][int(np.argwhere(np.ceil(cts_cum / min_cts) == i)[0])] * 1.
            if channels[-1] < max(data["r"]):
                channels = np.append(channels, [max(data["r"]) + 1e-3])

        channel_centroids = (channels[1:] + channels[:-1]) / 2
        channel_lowerrs = channel_centroids - channels[:-1]
        channel_uperrs = channels[1:] - channel_centroids
        # Start binning.
        channel_index = np.digitize(data["r"], channels)
        data = hstack([data, Table(np.atleast_2d(channel_index).T, dtype=[int], names=["index"])])
        grouped = data.group_by(channel_index)
        grouped = grouped.groups
        profile = Table(np.zeros([len(grouped), 10]),
                        dtype=[float, float, float, float, float, float, float, float, int, float],
                        names=["r", "r_uperr", "r_lowerr", "flux", "flux_err", "cts", "bkg_cts", "exp", "npix",
                               "scale"])
        scale = self.xscale * self.yscale * 3600  # Sky area per pixel (unit: arcmin^-2)
        n_files = len(self.ctsimages.data)
        for i in range(len(grouped)):
            subtable = grouped[i]
            cts = np.zeros(n_files)
            exps = np.zeros(n_files)
            bkg_cts = np.zeros(n_files)
            flux = np.zeros(n_files)
            flux_err = np.zeros(n_files)
            for j in range(len(self.ctsimages.data)):
                subsubtable = subtable[subtable["i"] == j]
                exps[j] = np.sum(subsubtable["exp"])
                cts[j] = np.sum(subsubtable["cts"])
                flux[j] = np.sum(subsubtable["net_cts"]) / exps[j] / scale
                bkg_cts[j] = np.sum(subsubtable["cts"]) - np.sum(subsubtable["net_cts"])
                # bkgnorm = bkg_cts[j] / np.sum(subsubtable["raw_bkg"])
                bkgnorm = np.nanmean(subsubtable["bkgnorm"])
                flux_err[j] = np.sqrt(
                    np.sum(subsubtable["cts"]) + np.sum(subsubtable["raw_bkg"]) * bkgnorm ** 2) / exps[j] / scale
            profile["cts"][i] = np.sum(cts)
            profile["bkg_cts"][i] = np.sum(bkg_cts)
            profile["exp"][i] = np.sum(subtable["exp"])
            profile["npix"][i] = len(subtable)
            profile["scale"][i] = scale
            # Weighted averaged flux and error.
            # profile["flux"][i], profile["flux_err"][i] = weighted_average(flux, flux_err)
            # Or
            profile["flux"][i] = (profile["cts"][i] - profile["bkg_cts"][i]) / profile["exp"][i] / profile["scale"][i]
            bkgnorm = profile["bkg_cts"][i] / np.sum(subtable["raw_bkg"])
            profile["flux_err"][i] = np.sqrt(profile["cts"][i] + np.sum(subtable["raw_bkg"]) * bkgnorm ** 2) / \
                                     profile["exp"][i] / profile["scale"][i]
            index = subtable["index"][0] - 1
            profile["r"][i] = channel_centroids[index]
            profile["r_lowerr"][i] = channel_lowerrs[index]
            profile["r_uperr"][i] = channel_uperrs[index]
        return profile
