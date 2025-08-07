import numpy as np
from scipy.special import erf, erfinv

# === Distribution definitions ===

class Lognorm:
    def __init__(self, mu, sigma, N=1.0, base=np.e):
        self.mu = mu
        self.sigma = sigma
        self.N = N
        self.base = base
        if base == np.e:
            self.log = np.log
        elif base == 10:
            self.log = np.log10
        else:
            self.log = lambda x: np.log(x) / np.log(base)
        self.median = mu
        self.mean = mu * np.exp(0.5 * sigma**2)

    def pdf(self, x):
        scaling = self.N / (np.sqrt(2.0 * np.pi) * self.log(self.sigma))
        exponent = ((self.log(x / self.mu)) ** 2) / (2.0 * (self.log(self.sigma)) ** 2)
        return (scaling / x) * np.exp(-exponent)

    def cdf(self, x):
        erf_arg = (self.log(x / self.mu)) / (np.sqrt(2.0) * self.log(self.sigma))
        return (self.N / 2.0) * (1.0 + erf(erf_arg))

    def stats(self):
        stats_dict = dict()
        stats_dict["mean_radius"] = self.mu * np.exp(0.5 * self.sigma**2)
        stats_dict["total_diameter"] = self.N * stats_dict["mean_radius"]
        stats_dict["total_surface_area"] = 4.0 * np.pi * self.moment(2.0)
        stats_dict["total_volume"] = (4.0 * np.pi / 3.0) * self.moment(3.0)
        stats_dict["mean_surface_area"] = stats_dict["total_surface_area"] / self.N
        stats_dict["mean_volume"] = stats_dict["total_volume"] / self.N
        stats_dict["effective_radius"] = stats_dict["total_volume"] / stats_dict["total_surface_area"]
        return stats_dict

    def moment(self, k):
        scaling = (self.mu**k) * self.N
        exponent = ((k**2) / 2.0) * (self.log(self.sigma)) ** 2
        return scaling * np.exp(exponent)

    def __repr__(self):
        return f"Lognorm | mu = {self.mu:.2e}, sigma = {self.sigma:.2e}, N = {self.N:.2e} |"


# === AerosolSpecies simplified ===

def AerosolSpecies(species, distribution, kappa, rho=None, mw=None, bins=None, r_min=None, r_max=None):
    aerosol = {}
    aerosol["species"] = species
    aerosol["kappa"] = kappa
    aerosol["rho"] = rho
    aerosol["mw"] = mw
    aerosol["distribution"] = distribution

    if isinstance(distribution, Lognorm):
        if bins is None:
            raise ValueError("Need to specify `bins` when using a Lognorm distribution")

        if not r_min:
            lr = np.log10(distribution.mu / (10.0 * distribution.sigma))
        else:
            lr = np.log10(r_min)
        if not r_max:
            rr = np.log10(distribution.mu * 10.0 * distribution.sigma)
        else:
            rr = np.log10(r_max)

        rs = np.logspace(lr, rr, num=bins + 1)
        mids = np.array([np.sqrt(a * b) for a, b in zip(rs[:-1], rs[1:])])
        Nis = np.array([
            0.5 * (b - a) * (distribution.pdf(a) + distribution.pdf(b))
            for a, b in zip(rs[:-1], rs[1:])
        ])
        r_drys = mids * 1e-6

        aerosol["rs"] = rs
        aerosol["r_drys"] = r_drys
        aerosol["Nis"] = Nis * 1e6  # convert cm^-3 to m^-3
        aerosol["total_N"] = np.sum(aerosol["Nis"])
        aerosol["nr"] = len(r_drys)
    else:
        raise ValueError("Only Lognorm distributions supported in this function.")

    return aerosol
