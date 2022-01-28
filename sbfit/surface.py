import warnings
import copy
import itertools
import collections

import astropy.cosmology
import numpy as np
from scipy import optimize
from scipy import stats
from matplotlib import pyplot as plt
from astropy.modeling import Model
from astropy.wcs import WCS
from astropy.cosmology import Planck18
from astropy.coordinates import SkyCoord
from astropy import convolution

from . import observation
from . import statistics
from . import utils
from .region import Panda


class Surface(object):

    def __init__(self, stacked_cts_image, stacked_exp_image, stacked_bkg_image, wcs):
        self._wcs: WCS = wcs
        self._redshift = None
        self._cosmology: astropy.cosmology.FlatLambdaCDM = Planck18
        self._pixel_size = np.abs(np.diag(self._wcs.pixel_scale_matrix)[0]) * 60  # arcmin
        self._pixel_area = self._pixel_size ** 2  # arcmin^2
        self._cts_image = stacked_cts_image / self._pixel_area
        self._exp_image = stacked_exp_image
        self._bkg_image = stacked_bkg_image / self._pixel_area  # todo for sanity
        self._flux_image = np.zeros_like(self._cts_image)
        self._model_cts_image = None
        self._model_flux_image = None
        self._residual_image = None
        self._raw_mask = self._exp_image > 0
        self._mask = self._exp_image > 0
        self._cts_image[~self._raw_mask] = 0
        self._bkg_image[~self._raw_mask] = 0
        self._calc_flux()
        self._model = None
        self._min_stat = None
        self._error_approx = None
        self._vorbin_number = None
        self._vorbin_xcoor = None
        self._vorbin_ycoor = None

    @property
    def redshift(self):
        return self._redshift

    @redshift.setter
    def redshift(self, z):
        if z > 0:
            self._redshift = z
        else:
            raise ValueError(f"Invalid input {z}. Redshift should be positive.")

    @property
    def model(self):
        if self._model is None:
            warnings.warn("No model set.")
        else:
            return self._model

    @property
    def cts_image(self):
        return self._cts_image

    @property
    def exp_image(self):
        return self._exp_image

    @property
    def bkg_image(self):
        return self._bkg_image

    @property
    def model_cts_image(self):
        return self._model_cts_image

    @property
    def model_flux_image(self):
        return self._model_flux_image

    @property
    def error(self):
        return self._error_approx

    def _calc_flux(self):
        flux_image = (self._cts_image - self._bkg_image) / self._exp_image
        flux_image[~self._raw_mask] = 0
        self._flux_image = flux_image

    def _calc_residual(self):
        if self._model_flux_image is None:
            raise Exception("Please set model first.")
        else:
            residual_image = self._flux_image / self._model_flux_image
            residual_image[~self._mask] = 0
            self._residual_image = residual_image

    def plot(self, image_type="flux", vmin=None, vmax=None):

        if image_type == "flux":
            image = self._flux_image
        elif image_type == "count":
            image = self._cts_image
        elif image_type == "exposure":
            image = self._exp_image
        elif image_type == "background":
            image = self._bkg_image
        elif image_type == "model":
            image = self._model_flux_image
        elif image_type == "residual":
            image = self._residual_image
        else:
            raise Exception("Keyword error. Only support "
                            "{flux, count, exposure, background, model, residual}")

        plt.figure()
        plt.imshow(image, origin="lower", vmin=vmin, vmax=vmax)
        plt.show()

    def set_redshift(self, z):
        self._redshift = z

    def set_model(self, model):
        if isinstance(model, Model):
            if model.n_inputs == 2:
                self._model = model
            else:
                raise Exception("Input model is not a 2D model.")
        else:
            raise TypeError("Input model is not a instance of astropy.modeling.Model.")

    def set_fit_radius(self, x, y, r, coord_type="image"):
        """
        Set a range for 2D fitting.

        Parameters
        ----------
        x : number
            Input X. Image or Sky coordinate.
        y : number
            Input Y. Image or Sky coordinate.
        r : number
            Radius for fit. If coord_type is "sky", then the unit is kpc.
        coord_type : {"image", "sky"}
            Coordinate type of the input. The 'sky' option is for FK5 frame.

        """
        ly, lx = self._cts_image.shape
        xcoor, ycoor = np.meshgrid(np.arange(lx), np.arange(ly), )
        if coord_type == "image":
            fit_mask = np.sqrt((xcoor - x) ** 2 + (ycoor - y) ** 2) <= r
        elif coord_type == "sky":
            if self._redshift is None:
                raise Exception("Set redshift first.")
            else:
                sky_coor = SkyCoord(x, y, frame="fk5", unit="degree")
                im_x, im_y = sky_coor.to_pixel(self._wcs)
                kpc_per_arcmin = self._cosmology.kpc_proper_per_arcmin(self._redshift).value
                kpc_per_pix = kpc_per_arcmin * self._pixel_size
                r /= kpc_per_pix
                fit_mask = np.sqrt((xcoor - im_x) ** 2 + (ycoor - im_y) ** 2) <= r
        else:
            raise Exception("coord_type should be in {'image', 'sky'}.")

        print(f"Set radius {int(r): d} pixels")
        self._mask = np.logical_and(fit_mask, self._raw_mask)

        # voronoi binning
        # print(f"Start voronoi binning.")
        # y_index, x_index = np.where(self._mask)
        # print(f"X range [{np.min(x_index)}, {np.max(x_index)}], "
        #       f"Y range [{np.min(y_index)}, {np.max(y_index)}].")
        # self._vorbin_number, self._vorbin_xcoor, self._vorbin_ycoor = \
        #     utils.voronoi(self._cts_image * self._pixel_area, np.min(x_index), np.max(x_index),
        #                   np.min(y_index), np.max(y_index), snr=None)
        # print(f"Finish voronoi binning.")

    def evaluate_pa(self, smooth_width=10, redshift=0.1, r_in=100, r_out=200):
        kernel = convolution.Gaussian2DKernel(smooth_width)
        smoothed_flux = convolution.convolve_fft(self._flux_image, kernel)
        cy, cx = np.unravel_index(smoothed_flux.argmax(), smoothed_flux.shape)
        kpc_per_pixel = self._pixel_size * Planck18.kpc_proper_per_arcmin(redshift).value
        panda_reg = Panda()
        panda_reg.frame = "image"
        panda_reg.set_parameters(x=cx + 1e-5, y=cy + 1e-5, startangle=0, stopangle=180,
                                 nangle=1, inner=r_in / kpc_per_pixel, outer=r_out / kpc_per_pixel)
        pa_coord, pa_mask = panda_reg.get_x_coordinate(self._flux_image,
                                                       header=self._wcs.to_header(), axis="y")
        # ycoor, xcoor = np.indices(smoothed_flux.shape)
        sum_flux, az_bin, _ = stats.binned_statistic(np.ravel(pa_coord[pa_mask]), np.ravel(smoothed_flux[pa_mask]),
                                                     statistic="sum", bins=np.arange(181) * 2)
        sum_flux = sum_flux.reshape([2, 90]).sum(axis=0)
        sum_flux = np.concatenate([sum_flux, sum_flux])
        az_bin = 0.5 * (az_bin[:90] + az_bin[1:91])
        az_bin = np.concatenate([az_bin - 180, az_bin])
        sum_flux = convolution.convolve(sum_flux, convolution.Gaussian1DKernel(2))
        pa = az_bin[sum_flux.argmax()]
        if pa < 0:
            pa += 180
        return pa

    def calculate(self, update=True):
        ysize, xsize = self._flux_image.shape
        xcoor, ycoor = np.meshgrid(np.arange(xsize), np.arange(ysize))
        model: Model = self.model
        model_flux = model.evaluate(xcoor, ycoor, *model.parameters)
        model_flux[np.logical_not(self._raw_mask)] = 0
        model_cts = model_flux * self.exp_image + self.bkg_image
        if update:
            self._model_flux_image = model_flux
            self._model_cts_image = model_cts
            self._calc_residual()
        else:
            pass

        # calculate c-stat
        cts_array = self.cts_image[self._mask].flatten()
        model_cts_array = model_cts[self._mask].flatten()
        stat_value = statistics.cstat(cts_array, model_cts_array)
        return stat_value

    def fit(self, show_step=True, tol=1e-2, nfail=10, show_result=True, record_fit=True):
        if self._model is None:
            raise Exception("Please set model first.")
        else:
            pass

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
            shift: np.matrix = np.linalg.pinv(
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
                if show_step:
                    print(f"C-stat: {new_stat:.3f}\n{new_pvalues_free}")
                else:
                    pass

        stat = self.calculate(update=True)

        if show_step:
            print("Iteration terminated.")
        errors = np.array(np.sqrt(
            np.abs(np.linalg.pinv(alpha).diagonal()))).flatten()

        pnames_free, pvalues_free = utils.get_free_parameter(self.model)
        dof = np.sum(self._mask) - len(pnames_free)
        if show_result:
            print(f"Degree of freedom: {dof:d}; C-stat: {stat:.4f}")
            for item in pnames_free:
                print(
                    f"{item}:\t"
                    f"{self.model.__getattribute__(item).value:.2e}")
            print(f"Uncertainties from rough estimation:")
            [print(f"{pnames_free[i]}:\t{errors[i]:.3e}") for i in
             range(len(pnames_free))]
        if record_fit:
            self._min_stat = stat
            self._error_approx = collections.OrderedDict(zip(pnames_free,
                                                             errors))
        return stat

    def _stat_deriv(self, pnames=None):
        """
        Calculate the first derivative of the statistic value versus each
        parameter.

        Parameters
        ----------
        pnames : list, optional
            If set, only calculate the first derivative versus the given
            parameter.
        Returns
        -------
        deriv : ndarray
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
            self._model = copy.deepcopy(current_model)
        deriv = np.array(deriv)
        return deriv

    def _stat_deriv_matrix(self):
        """
        To calculate the second derivative matrix of the statistic value.

        Returns
        -------
        deriv_matrix : matrix
            The calculated matrix.

        """
        shift = 1e-2
        pnames_free, pvalues_free = utils.get_free_parameter(self.model)
        current_model = copy.deepcopy(self.model)
        deriv_matrix = np.zeros([len(pnames_free), len(pnames_free)])
        for comb in itertools.combinations(np.arange(len(pnames_free)), 2):
            stat0 = self._stat_deriv(pnames=[pnames_free[comb[0]]])[0]
            self.model.__setattr__(pnames_free[comb[1]],
                                   pvalues_free[comb[1]] * (1 + shift) + 1e-10)
            stat1 = self._stat_deriv(pnames=[pnames_free[comb[0]]])[0]
            self._model = copy.deepcopy(current_model)
            second_deriv = (stat1 - stat0) / (
                    1e-10 + pvalues_free[comb[1]] * shift)
            deriv_matrix[comb] = second_deriv
        deriv_matrix += deriv_matrix.T
        for i in range(len(pnames_free)):
            stat0 = self._stat_deriv(pnames=[pnames_free[i]])[0]
            self.model.__setattr__(pnames_free[i],
                                   pvalues_free[i] * (1 + shift) + 1e-10)
            stat1 = self._stat_deriv(pnames=[pnames_free[i]])[0]
            self._model = copy.deepcopy(current_model)
            second_deriv = (stat1 - stat0) / (1e-10 + pvalues_free[i] * shift)
            deriv_matrix[i, i] = second_deriv
        return np.mat(deriv_matrix)

    def fit_alt(self):
        # if self._vorbin_number is None:
        #     raise Exception("Set fit radius first.")
        # else:
        #     pass
        pnames_free, pvalues_free = utils.get_free_parameter(self.model)
        p_bounds_low, p_bounds_high = utils.get_parameter_bounds(self.model, pnames_free)
        input_bound = np.array([p_bounds_low, p_bounds_high]).T.tolist()
        model = self._model.deepcopy()
        print(pvalues_free, input_bound)
        fit_result = optimize.minimize(self.fit_func, pvalues_free,
                                       method="Nelder-Mead",
                                       # method="Powell",
                                       args=(model, pnames_free, self._cts_image,
                                             self._exp_image, self._bkg_image, self._mask,
                                             # self._vorbin_xcoor, self._vorbin_ycoor,
                                             # self._vorbin_number
                                             ),
                                       bounds=input_bound, options={"disp": True, },
                                       )
        # fit_result = optimize.least_squares(self.fit_func, pvalues_free,
        #                                     method="trf", x_scale="jac", loss="linear",
        #                                     args=(model, pnames_free, self._cts_image,
        #                                           self._exp_image, self._bkg_image, self._mask),
        #                                     bounds=(p_bounds_low, p_bounds_high), verbose=2,
        #                                     )
        fit_result: optimize.OptimizeResult
        print(fit_result.x)
        for i, pvalue in enumerate(fit_result.x):
            self._model.__setattr__(pnames_free[i], pvalue)
        stat = self.calculate()
        # stat = self.vorbin_cstat(self._cts_image, self._model_cts_image, self._mask,
        #                          self._vorbin_xcoor, self._vorbin_ycoor, self._vorbin_number)
        dof = self._cts_image[self._mask].size - len(pnames_free)
        # dof = np.int(np.max(self._vorbin_number)) + 1 - len(pnames_free)
        print(f"C-stat / d.o.f = {stat} / {dof}")

    @staticmethod
    def fit_func_vorbin(params, model, param_names, cts_image, exp_image, bkg_image, mask,
                 xcoor_filter, ycoor_filter, bin_num):
        ly, lx = cts_image.shape
        xcoor, ycoor = np.meshgrid(np.arange(lx), np.arange(ly))
        for i in range(len(params)):
            model.__setattr__(param_names[i], params[i])
        model_image = model.evaluate(xcoor, ycoor, *model.parameters)
        model_cts_image = model_image * exp_image + bkg_image
        cts_image[~mask] = 0
        model_cts_image[~mask] = 0
        cts_array = utils.stat_with_index_2d(cts_image, xcoor_filter, ycoor_filter,
                                             bin_num, method="sum")
        model_cts_array = utils.stat_with_index_2d(model_cts_image, xcoor_filter, ycoor_filter,
                                                   bin_num, method="sum")
        stat = statistics.cstat(cts_array, model_cts_array)
        return stat

    @staticmethod
    def fit_func(params, model, param_names, cts_image, exp_image, bkg_image, mask):
        ly, lx = cts_image.shape
        xcoor, ycoor = np.meshgrid(np.arange(lx), np.arange(ly))
        for i in range(len(params)):
            model.__setattr__(param_names[i], params[i])
        model_image = model.evaluate(xcoor, ycoor, *model.parameters)
        model_cts_image = model_image * exp_image + bkg_image
        cts_image[~mask] = 0
        model_cts_image[~mask] = 0
        stat = statistics.cstat(cts_image[mask].flatten(), model_cts_image[mask].flatten())
        return stat

    @staticmethod
    def vorbin_cstat(cts_image, model_cts_image, mask, xcoor, ycoor, bin_num):
        cts_image[~mask] = 0
        model_cts_image[~mask] = 0
        cts_array = utils.stat_with_index_2d(cts_image, xcoor, ycoor,
                                             bin_num, method="sum")
        model_cts_array = utils.stat_with_index_2d(model_cts_image, xcoor, ycoor,
                                                   bin_num, method="sum")
        stat = statistics.cstat(cts_array, model_cts_array)
        return stat

    def save_image(self, filename, image_type="flux", smooth=False, width=1):
        """
        Save image to FITS file.

        Parameters
        ----------
        filename : str
            Output destination.
        image_type : {'flux', 'model_flux', 'model_cts', 'residual'}, optional
            Type of output. Default = 'flux'.
        smooth : bool
            Whether smooth the image before output. Default = False.
        width : number
            Gaussian smooth kernel width.
        """
        if image_type == "flux":
            out_image = self._flux_image * self._pixel_area
        elif image_type == "model_flux":
            out_image = self._model_flux_image * self._pixel_area
        elif image_type == "model_cts":
            out_image = self._model_cts_image * self._pixel_area
        elif image_type == "residual":
            out_image = self._residual_image
        if smooth:
            kernel = convolution.Gaussian2DKernel(width)
            con_image = convolution.convolve(out_image, kernel)
            con_image[np.logical_not(self._raw_mask)] = 0
            out_image = con_image

        header = self._wcs.to_header()
        utils.write_fits_image(out_image, filename, header=header)

    def save_model(self, filename):
        kernel = convolution.Gaussian2DKernel(25)
        con_exp = convolution.convolve_fft(self._exp_image, kernel)
        con_bkg = convolution.convolve_fft(self._bkg_image, kernel)
        res_cts = con_bkg / con_exp * self._exp_image + self._model_flux_image * self._exp_image
        res_cts[np.logical_not(self._raw_mask)] = 0
        res_cts *= self._pixel_area
        header = self._wcs.to_header()
        utils.write_fits_image(res_cts, filename, header=header)
