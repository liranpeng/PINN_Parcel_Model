#!/bin/bash
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -J parcel
#SBATCH -A m4334
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liranp@uci.edu
#SBATCH -L scratch,cfs
#SBATCH -C cpu
#SBATCH --ntasks=128

module load python
module load scipy-stack

# ✅ Fully self-contained parcel model with Ghan supersaturation equations
cat << 'EOF' > parcel_model_run.py
import os, math
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import bisect

############################################
# 1️⃣ CONSTANTS
############################################
g = 9.81
Cp = 1004.0
L = 2.5e6        # latent heat of vaporization [J/kg]
rho_w = 1000.0
R = 8.314
Mw = 18.0 / 1000.0   # kg/mol
Ma = 28.9 / 1000.0
Rd = R / Ma
Rv = R / Mw
Dv = 2.5e-5       # diffusivity of water vapor [m2/s]
Ka = 2.4e-2       # thermal conductivity of air [J/m/s/K]
ac = 1.0          # condensation coefficient
epsilon = 0.622

STATE_VARS = ["z", "P", "T", "wv", "wc", "wi", "S"]
STATE_VAR_MAP = {var: i for i, var in enumerate(STATE_VARS)}

############################################
# 2️⃣ SUPPORT FUNCTIONS
############################################
def es(T_c):
    """Saturation vapor pressure over water [Pa], T in °C."""
    return 610.94 * math.exp(17.625 * T_c / (T_c + 243.04))

def rho_air(T, P):
    return P / (Rd * T)

def Seq(r, r_dry, T, kappa):
    """Köhler equilibrium supersaturation."""
    sigma = 0.072
    A = 2 * sigma * Mw / (R * T * rho_w)
    B = kappa * r_dry**3
    return math.exp(A/r - B/(r**3)) - 1

def kohler_crit(T, r_dry, kappa):
    """Critical Köhler radius & supersat."""
    sigma = 0.072
    A = 2 * sigma * Mw / (R * T * rho_w)
    B = kappa * r_dry**3
    r_crit = math.sqrt(3*B/A)
    S_crit = math.exp(4*A/(27*B)) - 1
    return r_crit, S_crit

def condensational_growth_rate(r, T, S, P):
    """
    Compute dr/dt for a single droplet (Ghan eqns).
    """
    # saturation vapor pressure
    e_s = es(T-273.15)
    rho_a = rho_air(T, P)

    # terms for mass transfer
    G_v = (Rv * T) / (Dv * e_s)
    G_t = (L * Mw) / (Ka * R * T) * (L / (R * T) - 1)
    G = 1.0 / (rho_w * (G_v + G_t))

    # supersaturation driven growth (eq. 9)
    drdt = (G / r) * S
    return drdt

############################################
# 3️⃣ DISTRIBUTIONS & AEROSOLS
############################################
class Lognorm:
    def __init__(self, mu, sigma, N=1.0):
        self.mu = mu
        self.sigma = sigma
        self.N = N
    def pdf(self, x):
        scaling = self.N / (np.sqrt(2.0 * np.pi) * np.log(self.sigma))
        exponent = ((np.log(x / self.mu)) ** 2) / (2.0 * (np.log(self.sigma)) ** 2)
        return (scaling / x) * np.exp(-exponent)

class AerosolSpecies:
    def __init__(self, species, distribution, kappa, bins=50):
        self.species = species
        self.distribution = distribution
        self.kappa = kappa
        self.bins = bins

        # log-spaced bins
        lr = np.log10(distribution.mu / (10.0 * distribution.sigma))
        rr = np.log10(distribution.mu * 10.0 * distribution.sigma)
        self.rs = np.logspace(lr, rr, num=bins + 1)
        mids = np.array([np.sqrt(a * b) for a, b in zip(self.rs[:-1], self.rs[1:])])
        self.r_drys = mids * 1e-6
        self.Nis = np.array(
            [0.5 * (b - a) * (distribution.pdf(a) + distribution.pdf(b))
             for a, b in zip(self.rs[:-1], self.rs[1:])]
        ) * 1e6
        self.N = np.sum(self.Nis)

############################################
# 4️⃣ PARCEL MODEL with GHAN dS/dt
############################################
class ParcelModel:
    def __init__(self, aerosols, V, T0, S0, P0):
        self.aerosols = aerosols
        self.V = V
        self.T0 = T0
        self.S0 = S0
        self.P0 = P0
        self._setup_run()

    def _setup_run(self):
        """Set up equilibrium droplet radii and state vector."""
        self.z0 = 0.0
        self.wv0 = (self.S0 + 1.0) * (epsilon * es(self.T0 - 273.15) / (self.P0 - es(self.T0 - 273.15)))
        self.wc0 = 0.0
        self.wi0 = 0.0

        self._r0s = []
        self._r_drys = []
        self._Nis = []
        self._kappas = []

        for aerosol in self.aerosols:
            for r_dry, Ni in zip(aerosol.r_drys, aerosol.Nis):
                r_b, _ = kohler_crit(self.T0, r_dry, aerosol.kappa)
                f = lambda r: Seq(r, r_dry, self.T0, aerosol.kappa) - self.S0
                r0 = bisect(f, r_dry, r_b, xtol=1e-30, maxiter=500)
                self._r0s.append(r0)
                self._r_drys.append(r_dry)
                self._Nis.append(Ni)
                self._kappas.append(aerosol.kappa)

        self._r0s = np.array(self._r0s)
        self._r_drys = np.array(self._r_drys)
        self._Nis = np.array(self._Nis)
        self._kappas = np.array(self._kappas)

        self.y0 = np.concatenate(([self.z0, self.P0, self.T0,
                                   self.wv0, self.wc0, self.wi0, self.S0],
                                  self._r0s))

    def _rhs(self, y, t):
        """Right-hand side ODE with Ghan dS/dt."""
        z, P, T, wv, wc, wi, S = y[:7]
        radii = y[7:]

        # eq. 8: supersat tendency components
        rho_a = rho_air(T, P)
        alpha = (g * Mw * L) / (Cp * R * T**2) - (g * Ma) / (R * T)  # cooling term
        beta = (Rv * T) / (Dv * es(T-273.15)) + (L * Mw) / (Ka * R * T) * (L/(R*T)-1)
        G = 1.0 / (rho_w * beta)

        # compute droplet growth (eq. 9-11)
        drdt_all = []
        condensational_sink = 0.0

        for i, r in enumerate(radii):
            drdt = (G / r) * S
            drdt_all.append(drdt)
            condensational_sink += self._Nis[i] * r * drdt

        # eq. 8: supersaturation tendency
        dSdt = alpha * self.V - (Mw/(rho_a*Rv*T)) * condensational_sink

        # other tendencies (simplified for now)
        dzdt = self.V
        dPdt = -rho_a * g * self.V
        dTdt = -g/Cp * self.V

        return [dzdt, dPdt, dTdt, 0, 0, 0, dSdt] + drdt_all

    def run(self, t_end=60.0, output_dt=1.0):
        t = np.arange(0, t_end + output_dt, output_dt)
        sol = odeint(self._rhs, self.y0, t)
        out = pd.DataFrame(sol[:, :7], columns=STATE_VARS)
        return out

############################################
# 5️⃣ PARALLEL EXECUTION
############################################
rank = int(os.environ.get("SLURM_PROCID", 0))
np.random.seed(rank)

N_samples = 150
all_inputs = []
all_outputs = []

for i in range(N_samples):
    # 🎲 Random ICs
    P0 = np.random.uniform(70000, 90000)
    T0 = np.random.uniform(270, 290)
    S0 = np.random.uniform(-0.05, 0.05)
    V  = np.random.uniform(0.1, 2.0)
    mu = np.random.uniform(0.01, 0.03)
    sigma = np.random.uniform(1.4, 1.8)
    N = np.random.uniform(500, 1500)

    sulfate = AerosolSpecies('sulfate', Lognorm(mu=mu, sigma=sigma, N=N), kappa=0.54, bins=50)
    model = ParcelModel([sulfate], V, T0, S0, P0)

    try:
        par_out = model.run(t_end=60.0, output_dt=1.0)
    except Exception as e:
        print(f"[Rank {rank}] Simulation {i} failed: {e}")
        continue

    all_inputs.append({'P0': P0, 'T0': T0, 'S0': S0, 'V': V,
                       'mu': mu, 'sigma': sigma, 'N': N})
    all_outputs.append(par_out.to_dict(orient='list'))

np.savez_compressed(f"parcel_results_rank{rank}.npz",
                    inputs=all_inputs,
                    outputs=all_outputs)
EOF

# ✅ Run across 128 cores
srun -n 128 python parcel_model_run.py
