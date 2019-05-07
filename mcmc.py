import numpy as np
import emcee


def mcmc_err(profile, model, stat="C-Stat", nwalkers=100, nburnin=100, nstep=500):
    pass


class Sampler(object):

    def __init__(self, profile, model, stat="C-Stat"):
        self.profile = profile
        self.model = model
        self.stat = stat
        free_param_mask = np.logical_not(np.array(list(self.model.fixed.values())))
        theta_name = np.array(self.model.param_names)[free_param_mask]
        theta_value = self.model.parameters[free_param_mask]
        theta_bounds = np.array(list(self.model.bounds.values()))[free_param_mask]


    def lnprob(self):
        if self.stat == "C-Stat":
            pass
