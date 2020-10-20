import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.table import Table, vstack, hstack
from .region import Epanda, Panda, Circle
from .exception import *
from .utils import xy2elliptic, isincircle
from .image import CtsImageList, ExpImageList, BkgImageList


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
        if not (isinstance(self.add_region, Panda) or isinstance(self.add_region, Epanda)):
            raise TypeError("Add region must be Panda or Epanda")
        self.get_data()

    def get_data(self):
        """Collecting data from images"""
        add_region = self.add_region
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
                # Region coordinates transfer.
                x0, y0 = wcs.all_world2pix(np.array([[add_region.x.value, add_region.y.value]]), 1)[0]
                # FITS image coordinate order should start from 1.
                # Build grid of pixels.
                xscale, yscale = proj_plane_pixel_scales(wcs)
                self.xscale = xscale
                self.yscale = yscale
                outermajor = add_region.outermajor.value / 3600 / xscale  # Unit: pixel
                outerminor = add_region.outerminor.value / 3600 / xscale
                innermajor = add_region.innermajor.value / 3600 / xscale
                innerminor = add_region.innerminor.value / 3600 / xscale
                for sub_region in self.sub_region:
                    if isinstance(sub_region, Circle):
                        xsub, ysub = wcs.all_world2pix(np.array([[sub_region.x.value, sub_region.y.value]]), 1)[0]
                        rsub = sub_region.radius.value / 3600 / xscale
                        subtraction_mask = np.logical_or(isincircle(xcoor, ycoor, xsub, ysub, rsub), subtraction_mask)
                    else:
                        warnings.warn("So far only Circle subtraction region is supported.")
            else:
                # TODO Add features for non-celestial regions.
                pass
            az, r_pix = xy2elliptic(xcoor, ycoor, x0, y0, outermajor, outerminor, add_region.angle, startangle,
                                    stopangle)
            exist_mask = (self.expimages.data[i] * 1.) > 0
            exist_mask = np.logical_and(exist_mask, np.logical_not(subtraction_mask))
            # rad_mask = np.logical_and(r_pix >= innermajor, r_pix < outermajor)
            rad_mask = r_pix >= innermajor
            if stopangle > 360:
                az_mask = np.logical_or(az >= startangle, az < stopangle - 360)
            else:
                az_mask = np.logical_and(az >= startangle, az < stopangle)
            valid_mask = np.logical_and(exist_mask, np.logical_and(rad_mask, az_mask))
            r_pix_valid = r_pix[valid_mask]
            r_arcmin_valid = r_pix_valid * xscale * 60
            az_valid = az[valid_mask]
            # lx_valid = lx[valid_mask]
            # ly_valid = ly[valid_mask]
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
                Table(np.array([r_arcmin_valid, az_valid, cts_valid, exp_valid, bkg_valid, cts_valid, net_cts_valid,
                                i * np.ones(len(cts_valid))]).T,
                      names=("r", "az", "cts", "exp", "raw_bkg", "raw_cts", "net_cts", "i"),
                      dtype=(float, float, float, float, float, float, float, int))]
            self.data = vstack(sub_data)
            self.data.sort("r")

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
                channels[i] = data["r"][int(np.argwhere(np.floor(cts_cum / min_cts) == i)[0])] * 1.
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
                # print(bkg_cts[j], np.sum(subsubtable["raw_bkg"]))
                bkgnorm = bkg_cts[j] / np.sum(subsubtable["raw_bkg"])
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
