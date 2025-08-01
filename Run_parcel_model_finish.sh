#!/bin/bash
#SBATCH -N 1                        # 1 node
#SBATCH -q debug                    # debug queue (change to regular later)
#SBATCH -t 00:30:00                 # time limit
#SBATCH -J parcel                   # job name
#SBATCH -A m4334                    # allocation
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liranp@uci.edu
#SBATCH -L scratch,cfs
#SBATCH -C cpu                      # CPU nodes only
#SBATCH --ntasks=128                # 128 parallel tasks (1 per CPU core)

module load python
module load scipy-stack   # or conda activate your env

# ✅ Write the inline ParcelModel code to a file
cat << 'EOF' > parcel_model_run.py
import numpy as np
from scipy.optimize import bisect
from scipy.integrate import odeint

#######################################
# === CONSTANTS ===
#######################################
g       = 9.81        # gravity [m/s2]
R       = 8.314       # universal gas constant [J/mol/K]
Rd      = 287.058     # gas constant for dry air [J/kg/K]
Cp      = 1004.0      # specific heat of air [J/kg/K]
L       = 2.5e6       # latent heat of vaporization [J/kg]
rho_w   = 1000.0      # density of water [kg/m3]
epsilon = 0.622       # ratio of gas constants (Rd/Rv)
Mw      = 0.018015    # molar mass of water [kg/mol]
Ma      = 0.02897     # molar mass of dry air [kg/mol]
STATE_VAR_MAP = {"z":0,"P":1,"T":2,"wv":3,"wc":4,"wi":5,"S":6}
N_STATE_VARS = 7

#######################################
# === SUPPORTING CLASSES ===
#######################################
class Lognorm:
    """Simple lognormal aerosol distribution placeholder."""
    def __init__(self, mu, sigma, N):
        self.mu = mu
        self.sigma = sigma
        self.N = N

class AerosolSpecies:
    """Minimal aerosol species class for parcel model."""
    def __init__(self, species, distribution, bins=200, kappa=0.54):
        self.species = species
        self.kappa = kappa
        self.bins = bins
        self.mu = distribution.mu
        self.sigma = distribution.sigma
        self.N = distribution.N
        # dry radius bins (log spaced)
        self.r_drys = np.logspace(np.log10(self.mu/2), np.log10(self.mu*2), bins)
        # assume equal number per bin for simplicity
        self.Nis = np.ones(bins) * (self.N / bins)
        self.nr = bins

#######################################
# === PHYSICS HELPERS ===
#######################################
def es(Tc):
    """Saturation vapor pressure over liquid water [Pa]. Tc in Celsius."""
    return 610.94 * np.exp(17.625*Tc/(Tc+243.04))

def rho_air(T, P, wv):
    """Air density accounting for humidity."""
    Tv = (1.0 + 0.61*wv) * T
    return P / (Rd * Tv)

def Seq(r, r_dry, T, kappa):
    """Equilibrium supersaturation over droplet."""
    return (1 + kappa * (r_dry/r)**3) * np.exp((2*0.072)/(rho_w*R*T*r*1e6)) - 1.0

def kohler_crit(T, r_dry, kappa):
    """Köhler critical radius and supersaturation (simplified placeholder)."""
    Scrit = 0.01  # dummy value
    rcrit = r_dry * (1 + kappa)**(1/3)
    return rcrit, Scrit

def dv(T, r, P, accom):
    """Vapor diffusivity [m2/s] placeholder."""
    return 2.26e-5

def ka(T, rho_air, r):
    """Thermal conductivity [W/m/K] placeholder."""
    return 2.4e-2

#######################################
# === ODE SYSTEM ===
#######################################
def parcel_ode_sys(y, t, nr, r_drys, Nis, V, kappas, accom=1.0):
    """
    Calculates time derivatives of parcel state variables.
    y[0:7] = z, P, T, wv, wc, wi, S
    y[7:]  = aerosol radii
    """
    z, P, T, wv, wc, wi, S = y[0:7]
    rs = np.asarray(y[N_STATE_VARS:])

    # Thermodynamics
    T_c = T - 273.15
    pv_sat = es(T_c)
    Tv = (1.0 + 0.61*wv) * T
    e = (1.0 + S) * pv_sat
    rho = P / (Rd * Tv)
    rho_dry = (P - e) / (Rd * T)

    # 1) Pressure
    dP_dt = -rho * g * V

    # 2) Droplet growth
    drs_dt = np.zeros(nr)
    dwc_dt = 0.0
    for i in range(nr):
        r = rs[i]
        r_dry = r_drys[i]
        kappa = kappas[i]
        # simplified growth rate
        G = 1.0
        delta_S = S - Seq(r, r_dry, T, kappa)
        dr_dt = (G/r) * delta_S
        Ni = Nis[i]
        dwc_dt += Ni * (r*r) * dr_dt
        drs_dt[i] = dr_dt

    dwc_dt *= 4.0 * np.pi * rho_w / rho_dry

    # 3) Ice water (not used here)
    dwi_dt = 0.0

    # 4) Water vapor
    dwv_dt = - (dwc_dt + dwi_dt)

    # 5) Temperature
    dT_dt = -g*V/Cp - L*dwv_dt/Cp

    # 6) Supersaturation
    alpha = (g*Mw*L)/(Cp*R*(T**2)) - (g*Ma)/(R*T)
    gamma = (P*Ma)/(Mw*es(T_c)) + (Mw*L*L)/(Cp*R*T*T)
    dS_dt = alpha*V - gamma*dwc_dt

    dz_dt = V

    dydt = np.zeros(nr + N_STATE_VARS)
    dydt[0] = dz_dt
    dydt[1] = dP_dt
    dydt[2] = dT_dt
    dydt[3] = dwv_dt
    dydt[4] = dwc_dt
    dydt[5] = dwi_dt
    dydt[6] = dS_dt
    dydt[N_STATE_VARS:] = drs_dt[:]
    return dydt

#######################################
# === PARCEL MODEL CLASS ===
#######################################
class ParcelModel:
    """
    Simplified Parcel Model stripped to essentials.
    """
    def __init__(self, aerosols, V, T0, S0, P0):
        self.aerosols = aerosols
        self.V = V
        self.T0 = T0
        self.S0 = S0
        self.P0 = P0
        self._setup_run()

    def _setup_run(self):
        r_drys, kappas, Nis = [], [], []
        for aerosol in self.aerosols:
            r_drys.extend(aerosol.r_drys)
            kappas.extend([aerosol.kappa]*aerosol.nr)
            Nis.extend(aerosol.Nis)

        self._r_drys = np.array(r_drys)
        self._kappas = np.array(kappas)
        self._Nis = np.array(Nis)
        self._nr = len(r_drys)

        # Initial state vector
        T0, S0, P0 = self.T0, self.S0, self.P0
        wv0 = (S0 + 1.0) * (epsilon * es(T0-273.15) / (P0 - es(T0-273.15)))
        wc0 = 0.0
        wi0 = 0.0
        r0s = np.array(self._r_drys)
        y0 = np.concatenate(([0.0, P0, T0, wv0, wc0, wi0, S0], r0s))
        self.y0 = y0

    def run(self, t_end, output_dt=1.0):
        t_grid = np.arange(0, t_end, output_dt)
        y_out = odeint(parcel_ode_sys, self.y0, t_grid,
                       args=(self._nr, self._r_drys, self._Nis, self.V, self._kappas))
        return t_grid, y_out

#######################################
# === MAIN EXECUTION ===
#######################################
if __name__ == "__main__":
    import os
    rank = int(os.environ.get("SLURM_PROCID", 0))  # MPI task index
    np.random.seed(rank)
    N_samples = 150

    all_results = []
    for i in range(N_samples):
        # Random initial conditions
        P0 = np.random.uniform(70000, 90000)   # Pa
        T0 = np.random.uniform(270, 290)       # K
        S0 = np.random.uniform(-0.05, 0.05)    # supersaturation
        V  = np.random.uniform(0.1, 2.0)       # updraft [m/s]
        mu = np.random.uniform(0.01, 0.03)     # microns
        sigma = np.random.uniform(1.4, 1.8)
        N = np.random.uniform(500, 1500)       # cm^-3

        aerosol = AerosolSpecies("sulfate", Lognorm(mu, sigma, N))
        model = ParcelModel([aerosol], V, T0, S0, P0)
        t, y = model.run(300, output_dt=1.0)
        S_profile = y[:,6]  # supersaturation time series
        all_results.append(S_profile)

    all_results = np.array(all_results)
    np.save(f"parcel_task_{rank}.npy", all_results)
EOF

# ✅ Launch 128 Python tasks (one per CPU core)
srun -n 128 python parcel_model_run.py

# ✅ Merge outputs on task 0 after finishing
if [[ $SLURM_PROCID -eq 0 ]]; then
python << 'PYEOF'
import numpy as np, glob
files = sorted(glob.glob("parcel_task_*.npy"))
all_data = [np.load(f) for f in files]
combined = np.concatenate(all_data, axis=0)
np.save("parcel_all.npy", combined)
print("✅ Combined dataset saved. Shape:", combined.shape)
PYEOF
fi

