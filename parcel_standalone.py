# parcel_standalone.py
# Standalone, pyrcel-style parcel model with full physics ODE + CVODE/SciPy integrators

import sys, time, warnings
import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.optimize import bisect, fminbound

# ----------------------------
# Constants (match pyrcel ones)
# ----------------------------
g = 9.81                  # m s^-2
Cp = 1004.0               # J/(kg K)
L = 2.25e6                # J/kg
rho_w = 1e3               # kg/m^3
R = 8.314                 # J/(mol K)
Mw = 18.0 / 1e3           # kg/mol
Ma = 28.9 / 1e3           # kg/mol
Rd = R / Ma               # J/(kg K)
Rv = R / Mw               # J/(kg K)
Dv = 3.0e-5               # m^2/s  (base reference; we’ll compute effective dv)
Ka = 2.0e-2               # J/(m s K) (base reference; we’ll compute effective ka)
ac_default = 1.0          # condensation (mass accommodation) coefficient
at = 0.96                 # thermal accommodation coefficient
epsilon = 0.622           # Mw / Ma (≈ Rd/Rv)

PI = np.pi
N_STATE_VARS = 7          # [z, P, T, wv, wc, wi, S]
Z_IDX, P_IDX, T_IDX, S_IDX = 0, 1, 2, 6

# ---------------------------------
# Minimal aerosol helpers/classes
# ---------------------------------
class Lognorm:
    """
    Lognormal distribution descriptor for binning convenience.
    mu, sigma in microns; N in #/cm^3 (or any consistent unit you use).
    """
    def __init__(self, mu, sigma, N):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.N = float(N)

    def __repr__(self):
        return f"Lognorm | mu = {self.mu:.2e}, sigma = {self.sigma:.2e}, N = {self.N:.2e} |"


class AerosolSpecies:
    """
    Discretize a lognormal into logarithmically spaced rs (wet centers used only for display),
    and generate r_drys + Nis per bin. This mirrors pyrcel’s prepared bins output.
    Units: we’ll keep radii in meters inside the model.
    """
    def __init__(self, species, distribution: Lognorm, kappa=0.0, bins=200,
                 r_min=None, r_max=None):
        self.species = species
        self.distribution = distribution
        self.kappa = float(kappa)
        self.bins = int(bins)

        # center mu is microns (per many aerosol configs) -> convert to meters
        mu_m = distribution.mu * 1e-6
        # default bin range: mu/32 .. mu*32; allow override
        r_lo = (mu_m/32.0) if r_min is None else float(r_min)
        r_hi = (mu_m*32.0) if r_max is None else float(r_max)
        edges = np.geomspace(max(r_lo, 1e-10), max(r_hi, r_lo*1.0001), bins+1)
        centers = np.sqrt(edges[:-1]*edges[1:])

        # number per bin: simple equal area in log-space scaled to total N
        # Here we just split equally; you can replace with proper lognormal
        Nis = np.full(bins, distribution.N / bins, dtype=float)

        self.rs = centers
        self.r_drys = centers.copy()     # start dry radius = center (user can pass their own bins)
        self.Nis = Nis                   # concentration per bin
        self.nr = bins
        self.total_N = float(np.sum(Nis))

    def summary_str(self):
        return (f"{self.distribution}, kappa={self.kappa}, bins={self.bins}, "
                f"total_N={self.total_N:.3e}")


# -----------------------
# Thermo helper routines
# -----------------------
def sigma_w(T):
    """Surface tension of water [N/m], crude fit near 0–30C."""
    return 0.0761 - (1.55e-4) * (T - 273.15)

def ka_eff(T, r, rho_air):
    """
    Effective thermal conductivity of air with gas-kinetic correction.
    Mirrors the structure used in pyrcel (Fuchs–Sutugin style).
    """
    ka_cont = 1e-3 * (4.39 + 0.071 * T)  # W/(m K) == J/(m s K)
    denom = 1.0 + (ka_cont / (at * max(r, 1e-30) * rho_air * Cp)) * np.sqrt((2 * PI * Ma) / (R * T))
    return ka_cont / denom

def dv_eff(T, r, P, accom):
    """
    Effective vapor diffusivity with kinetic correction.
    """
    P_atm = max(P, 1.0) * 1.01325e-5  # Pa->atm; avoid zero
    dv_cont = 1e-4 * (0.211 / P_atm) * ((T / 273.0) ** 1.94)  # m^2/s
    denom = 1.0 + (dv_cont / (max(accom,1e-12) * max(r,1e-30))) * np.sqrt((2 * PI * Mw) / (R * T))
    return dv_cont / denom

def es(Tc):
    """Saturation vapor pressure (Pa) over liquid water; temperature in °C."""
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))

def rho_air(T, P, wv=0.0):
    """Moist air density (kg/m^3) via virtual temperature."""
    Tv = T * (1.0 + 0.61 * wv)
    return P / (Rd * Tv)

# Köhler equilibrium S_eq (Kelvin + Raoult) — scalar-safe
def Seq(r, r_dry, T, kappa):
    """
    Equilibrium supersaturation (fraction) at wet radius r for dry core r_dry and kappa.
    """
    r = float(r); r_dry = float(r_dry); kappa = float(kappa)
    # Kelvin term
    A = (2.0 * Mw * sigma_w(T)) / (R * T * rho_w * max(r, 1e-30))
    A = np.clip(A, -100.0, 100.0)
    # Raoult term (kappa-Köhler)
    if kappa > 0.0:
        r3, rd3 = r**3, r_dry**3
        den = r3 - rd3 * (1.0 - kappa)
        if den <= 0.0:
            den = 1e-30
        B = (r3 - rd3) / den
    else:
        B = 1.0
    return float(np.exp(A) * B - 1.0)

def kohler_crit(T, r_dry, kappa, approx=False):
    """
    Maximize Seq to find critical radius & supersaturation. Uses fminbound on -Seq.
    """
    r_dry = float(r_dry); kappa = float(kappa)
    if r_dry <= 0.0:
        raise ValueError("r_dry must be > 0")

    if approx or kappa <= 0.0:
        r_crit = r_dry * 2.0
        return r_crit, Seq(r_crit, r_dry, T, kappa)

    a = r_dry * (1.0 + 1e-6)
    b = r_dry * 1e4

    def neg_seq(r):
        return -Seq(r, r_dry, T, kappa)

    r_crit = float(fminbound(neg_seq, a, b, xtol=1e-10, maxfun=500, disp=0))
    S_crit = Seq(r_crit, r_dry, T, kappa)
    return r_crit, S_crit

# ----------------------------------------------------------
# Flatten aerosols (single dict/object OR list) -> bin arrays
# ----------------------------------------------------------
def flatten_aerosols(aerosols):
    """
    Accepts single AerosolSpecies/dict or list of them; returns concatenated arrays
    and metadata (bin counts per species + names).
    """
    if aerosols is None:
        raise ValueError("aerosols must be provided")

    if isinstance(aerosols, dict) or hasattr(aerosols, "r_drys"):
        aerosols_list = [aerosols]
    else:
        aerosols_list = list(aerosols)

    r_drys_all, Nis_all, kappas_all = [], [], []
    lengths, names = [], []

    for idx, aer in enumerate(aerosols_list, 1):
        if isinstance(aer, dict):
            rdi = np.asarray(aer["r_drys"], float).ravel()
            Nii = np.asarray(aer["Nis"], float).ravel()
            kapi = float(aer["kappa"])
            name = aer.get("species", f"species_{idx}")
        else:
            rdi = np.asarray(aer.r_drys, float).ravel()
            Nii = np.asarray(aer.Nis, float).ravel()
            kapi = float(aer.kappa)
            name = getattr(aer, "species", f"species_{idx}")

        if rdi.size != Nii.size:
            raise ValueError(f"Length mismatch in '{name}': r_drys({rdi.size}) vs Nis({Nii.size})")

        r_drys_all.append(rdi)
        Nis_all.append(Nii)
        kappas_all.append(np.full(rdi.shape, kapi, float))
        lengths.append(rdi.size)
        names.append(name)

    r_drys = np.concatenate(r_drys_all)
    Nis    = np.concatenate(Nis_all)
    kappas = np.concatenate(kappas_all)
    return r_drys, Nis, kappas, lengths, names

# ------------------------------------------------
# Build initial state y0 (pyrcel-style) from bins
# ------------------------------------------------
def build_initial_state(aerosols, V, T0, S0, P0):
    """
    Make initial y0 and return (y0, r_drys, Nis, kappas, nr, names).
    """
    r_drys, Nis, kappas, lengths, names = flatten_aerosols(aerosols)
    nr = int(r_drys.size)
    if nr == 0:
        raise ValueError("No aerosol bins found.")

    # initial water vapor mixing ratio (kg/kg)
    pv = es(T0 - 273.15)
    wv0 = (S0 + 1.0) * (epsilon * pv / (P0 - pv))

    # equilibrium wet radii r0s (solve Seq(r)=S0)
    r0s = []
    def f_r(r, rd, k):
        return Seq(r, rd, T0, k) - S0

    for rd, k in zip(reversed(r_drys), reversed(kappas)):
        rd = float(rd); k = float(k)
        r_b, _ = kohler_crit(T0, rd, k)
        r_a = rd * (1.0 + 1e-12)
        fa = f_r(r_a, rd, k); fb = f_r(r_b, rd, k)
        expand = 0
        while np.sign(fa) == np.sign(fb) and expand < 6:
            r_b *= 2.0
            fb = f_r(r_b, rd, k)
            expand += 1
        try:
            r0 = bisect(lambda r: f_r(r, rd, k), r_a, r_b, xtol=1e-30, maxiter=500)
        except ValueError:
            r0 = r_a
        r0s.append(float(r0))

    r0s = np.array(r0s[::-1], dtype=float)

    # initial cloud liquid water
    def water_vol(r0, rd, Ni):
        return (4.0 * PI / 3.0) * rho_w * Ni * (r0**3 - rd**3)

    wc0 = np.sum([water_vol(r0, rd, Ni) for r0, rd, Ni in zip(r0s, r_drys, Nis)])
    wc0 /= rho_air(T0, P0, 0.0)

    wi0 = 0.0
    z0 = 0.0
    y0 = np.array([z0, P0, T0, wv0, wc0, wi0, S0] + list(r0s), dtype=float)

    # sanity: bins shrinking although S0>=0
    if S0 >= 0.0:
        n_shrink = int(np.sum(r0s < r_drys))
        if n_shrink > 0:
            print(f"⚠️ {n_shrink} bins have r0 < r_dry while S0 >= 0.")

    return y0, r_drys, Nis, kappas, nr, names

# -------------------------------------------
# pyrcel-style parcel ODE (full microphysics)
# -------------------------------------------
def parcel_ode_sys(y, t, nr, r_drys, Nis, V, kappas, accom=ac_default):
    """
    y = [z, P, T, wv, wc, wi, S, r1..rN] ; returns dy/dt with same length.
    Implements the same physics as pyrcel’s accelerated version.
    """
    z, P, T, wv, wc, wi, S = y[:N_STATE_VARS]
    rs = np.asarray(y[N_STATE_VARS:])   # wet radii per bin

    T_c = T - 273.15
    pv_sat = es(T_c)                    # Pa
    Tv = (1.0 + 0.61 * wv) * T
    e = (1.0 + S) * pv_sat
    rho_air_now = P / (Rd * Tv)
    rho_air_dry = max((P - e) / (Rd * T), 1e-12)

    V_now = V(t) if callable(V) else V

    # 1) Pressure tendency
    dP_dt = -rho_air_now * g * V_now

    # 2/3) Wet growth + cloud liquid
    drs_dt = np.zeros(nr, dtype=float)
    dwc_dt = 0.0

    for i in range(nr):
        r = max(rs[i], 1e-30)
        r_dry = r_drys[i]
        kappa = kappas[i]
        Ni = Nis[i]

        dv_r = dv_eff(T, r, P, accom)
        ka_r = ka_eff(T, r, rho_air_now)

        # resistances (condensation + thermal)
        G_a = (rho_w * R * T) / (pv_sat * dv_r * Mw)
        G_b = (L * rho_w * ((L * Mw / (R * T)) - 1.0)) / (ka_r * T)
        G = 1.0 / (G_a + G_b)

        delta_S = S - Seq(r, r_dry, T, kappa)
        dr_dt = (G / r) * delta_S

        dwc_dt += Ni * (r * r) * dr_dt
        drs_dt[i] = dr_dt

    dwc_dt *= 4.0 * PI * rho_w / rho_air_dry

    # 4) Ice (off in this simple liquid-only config)
    dwi_dt = 0.0

    # 5) Water vapor
    dwv_dt = -1.0 * (dwc_dt + dwi_dt)

    # 6) Temperature
    dT_dt = -g * V_now / Cp - L * dwv_dt / Cp

    # 7) Supersaturation
    alpha = (g * Mw * L) / (Cp * R * (T**2)) - (g * Ma) / (R * T)
    gamma = (P * Ma) / (Mw * pv_sat) + (Mw * L * L) / (Cp * R * T**2)
    dS_dt = alpha * V_now - gamma * dwc_dt

    dz_dt = V_now

    # Pack
    dydt = np.zeros(N_STATE_VARS + nr, dtype=float)
    dydt[0] = dz_dt
    dydt[1] = dP_dt
    dydt[2] = dT_dt
    dydt[3] = dwv_dt
    dydt[4] = dwc_dt
    dydt[5] = dwi_dt
    dydt[6] = dS_dt
    dydt[N_STATE_VARS:] = drs_dt
    return dydt

# --------------------------------
# Integrators (CVODE / SciPy)
# --------------------------------
timer = time.process_time if sys.version_info[0] >= 3 else time.clock
state_atol = [1e-4, 1e-4, 1e-4, 1e-10, 1e-10, 1e-4, 1e-8]  # for [z,P,T,wv,wc,wi,S]
state_rtol = 1e-7

_have_cvode = False
try:
    from assimulo.exception import TimeLimitExceeded
    from assimulo.problem import Explicit_Problem
    from assimulo.solvers.sundials import CVode, CVodeError
    _have_cvode = True
except Exception:
    warnings.warn("Assimulo/SUNDIALS not available; will use SciPy if present.")

_have_scipy = False
try:
    from scipy.integrate import solve_ivp
    _have_scipy = True
except Exception:
    warnings.warn("SciPy not available. Only CVODE will work if installed.")

class Integrator(metaclass=ABCMeta):
    def __init__(self, rhs, output_dt, solver_dt, y0, args, t0=0.0, console=False):
        self.output_dt = float(output_dt)
        self.solver_dt = float(solver_dt)
        self.y0 = np.asarray(y0, dtype=float)
        self.t0 = float(t0)
        self.console = bool(console)
        self.args = args

        def _user_rhs(t, y):
            return rhs(y, t, *self.args)
        self.rhs = _user_rhs

    @abstractmethod
    def integrate(self, t_end, **kwargs):
        pass

    @staticmethod
    def solver(method="auto"):
        method = (method or "auto").lower()
        if method == "auto":
            if _have_cvode:
                return CVODEIntegrator
            if _have_scipy:
                return SciPyIntegrator
            raise ImportError("No integrator available: need Assimulo or SciPy.")
        if method == "cvode":
            if not _have_cvode:
                raise ImportError("CVODE requested but Assimulo/SUNDIALS not available.")
            return CVODEIntegrator
        if method in ("scipy", "odeint", "solve_ivp"):
            if not _have_scipy:
                raise ImportError("SciPy requested but not available.")
            return SciPyIntegrator
        raise ValueError(f"Unknown solver '{method}'")

# --- CVODE
if _have_cvode:
    class ExtendedProblem(Explicit_Problem):
        name = "Parcel model ODEs"
        sw0 = [True, False]   # before/after Smax
        t_cutoff = 1e5
        dS_dt = 1.0

        def __init__(self, rhs_fcn, rhs_args, terminate_depth, *args, **kwargs):
            self.rhs_fcn = rhs_fcn
            self.rhs_args = rhs_args
            V = rhs_args[3]
            V0 = (V(0.0) if callable(V) else float(V))
            self.terminate_time = float(terminate_depth) / max(V0, 1e-12)
            super(Explicit_Problem, self).__init__(*args, **kwargs)

        def rhs(self, t, y, sw):
            dode_dt = self.rhs_fcn(t, y)
            self.dS_dt = dode_dt[S_IDX]
            if sw[1]:  # past cutoff
                dode_dt = np.zeros(N_STATE_VARS + int(self.rhs_args[0]))
            return dode_dt

        def state_events(self, t, y, sw):
            smax_event = self.dS_dt if sw[0] else -1.0
            t_cutoff_event = t - self.t_cutoff
            return np.array([smax_event > 0, t_cutoff_event < 0])

        def handle_event(self, solver, event_info):
            event_info = event_info[0]
            if event_info[0] != 0:
                solver.sw[0] = False
                self.t_cutoff = solver.t + self.terminate_time

        def handle_result(self, solver, t, y):
            if t < self.t_cutoff:
                Explicit_Problem.handle_result(self, solver, t, y)

    class CVODEIntegrator(Integrator):
        def __init__(self, rhs, output_dt, solver_dt, y0, args, t0=0.0, console=False,
                     terminate=False, terminate_depth=100.0, **kwargs):
            self.terminate = bool(terminate)
            super().__init__(rhs, output_dt, solver_dt, y0, args, t0, console)

            if self.terminate:
                self.prob = ExtendedProblem(self.rhs, self.args, terminate_depth, y0=self.y0)
            else:
                self.prob = Explicit_Problem(self.rhs, self.y0)

            self.sim = self._setup_sim(**kwargs)

        def _setup_sim(self, **kwargs):
            sim = CVode(self.prob)
            sim.discr = "BDF"
            sim.maxord = 5
            sim.maxh = kwargs.get("maxh", min(0.1, self.output_dt))
            if "minh" in kwargs: sim.minh = kwargs["minh"]
            sim.iter = kwargs.get("iter", "Newton")
            if "linear_solver" in kwargs: sim.linear_solver = kwargs["linear_solver"]
            sim.maxsteps = kwargs.get("max_steps", 1000)
            sim.time_limit = kwargs.get("time_limit", 0.0)
            if sim.time_limit > 0: sim.report_continuously = True
            sim.store_event_points = False

            nr = int(self.args[0])
            sim.rtol = state_rtol
            sim.atol = state_atol + [1e-12] * nr
            sim.verbosity = 40 if self.console else 50
            return sim

        def integrate(self, t_end, **kwargs):
            t_increment = float(self.solver_dt)
            n_out = max(1, int(round(self.solver_dt / self.output_dt)))
            t_current = float(self.t0)

            if self.console:
                print("\nIntegration Loop (CVODE)\n")
                print("  step     time  walltime  Δwalltime |     z       T       S")
                print("--------------------------------------|----------------------")
                step_fmt = " {:5d} {:7.2f}s  {:7.2f}s  {:8.2f}s | {:5.1f} {:7.2f} {:6.2f}%"

            txs, xxs = [], []
            n_steps = 1
            total_walltime, now = 0.0, timer()

            while t_current < t_end:
                if self.console:
                    delta_walltime = timer() - now
                    total_walltime += delta_walltime
                    state = self.y0 if n_steps == 1 else xxs[-1][-1]
                    _z, _T, _S = state[Z_IDX], state[T_IDX], state[S_IDX] * 100.0
                    print(step_fmt.format(n_steps, t_current, total_walltime, delta_walltime, _z, _T, _S))
                try:
                    now = timer()
                    out_list = np.linspace(t_current, t_current + t_increment, n_out + 1)
                    tx, xx = self.sim.simulate(t_current + t_increment, 0, out_list)
                except (CVodeError, TimeLimitExceeded) as e:
                    raise ValueError(f"CVODE integration failed: {e}")

                if n_out == 1:
                    txs.append(tx[-1]); xxs.append(xx[-1])
                else:
                    txs.extend(tx[:-1]); xxs.append(xx[:-1])
                t_current = tx[-1]

                if self.terminate and not self.sim.sw[0]:
                    if self.console: print("---- termination condition reached ----")
                    break
                n_steps += 1

            if self.console: print("---- end of integration loop ----")
            t = np.array(txs)
            x = np.array(xxs) if n_out == 1 else np.concatenate(xxs)
            return x, t, True

        def __repr__(self):
            return "CVODE integrator - direct Assimulo interface"

# --- SciPy fallback
if _have_scipy:
    class SciPyIntegrator(Integrator):
        def __init__(self, rhs, output_dt, solver_dt, y0, args, t0=0.0, console=False,
                     terminate=False, terminate_depth=100.0, method="LSODA", **kwargs):
            self.terminate = bool(terminate)
            self.terminate_depth = float(terminate_depth)
            self.method = method
            super().__init__(rhs, output_dt, solver_dt, y0, args, t0, console)

            V = args[3]
            V0 = (V(0.0) if callable(V) else float(V))
            self.extra_time_after_smax = self.terminate_depth / max(V0, 1e-12)

        def _event_smax(self, t, y):
            return self.rhs(t, y)[S_IDX]
        _event_smax.terminal = True
        _event_smax.direction = 0.0

        def integrate(self, t_end, **kwargs):
            t_current = float(self.t0)
            y_current = self.y0.copy()
            t_vals, x_vals = [], []

            if self.console:
                print("\nIntegration Loop (SciPy/solve_ivp)\n")
                print("  step     time  walltime  Δwalltime |     z       T       S")
                print("--------------------------------------|----------------------")
                step_fmt = " {:5d} {:7.2f}s  {:7.2f}s  {:8.2f}s | {:5.1f} {:7.2f} {:6.2f}%"

            step, total_wall = 0, 0.0
            while t_current < t_end:
                step += 1
                now = timer()
                t_next = min(t_current + self.solver_dt, t_end)
                t_eval = np.arange(t_current, t_next + 1e-12, self.output_dt)
                events = [self._event_smax] if self.terminate else None

                sol = solve_ivp(
                    fun=self.rhs, t_span=(t_current, t_next), y0=y_current,
                    method=self.method, t_eval=t_eval,
                    rtol=state_rtol,
                    atol=np.array(state_atol + [1e-12] * int(self.args[0])),
                    events=events
                )
                if not sol.success:
                    raise ValueError(f"solve_ivp failed: {sol.message}")

                if len(sol.t) > 0:
                    if len(t_vals) and np.isclose(sol.t[0], t_vals[-1]):
                        t_vals.extend(sol.t[1:].tolist())
                        x_vals.extend(sol.y.T[1:].tolist())
                    else:
                        t_vals.extend(sol.t.tolist())
                        x_vals.extend(sol.y.T.tolist())

                if self.console:
                    delta_wall = timer() - now
                    total_wall += delta_wall
                    _z, _T, _S = x_vals[-1][Z_IDX], x_vals[-1][T_IDX], x_vals[-1][S_IDX] * 100.0
                    print(step_fmt.format(step, t_current, total_wall, delta_wall, _z, _T, _S))

                t_current = t_vals[-1]
                y_current = np.array(x_vals[-1])

                if self.terminate and sol.status == 1:  # dS/dt zero detected
                    t_extra_end = min(t_current + self.extra_time_after_smax, t_end)
                    if t_extra_end > t_current + 1e-9:
                        t_eval = np.arange(t_current, t_extra_end + 1e-12, self.output_dt)
                        sol2 = solve_ivp(
                            fun=self.rhs, t_span=(t_current, t_extra_end), y0=y_current,
                            method=self.method, t_eval=t_eval,
                            rtol=state_rtol,
                            atol=np.array(state_atol + [1e-12] * int(self.args[0])),
                        )
                        if not sol2.success:
                            raise ValueError(f"solve_ivp (post-terminate) failed: {sol2.message}")
                        if len(sol2.t) > 0:
                            if np.isclose(sol2.t[0], t_vals[-1]):
                                t_vals.extend(sol2.t[1:].tolist())
                                x_vals.extend(sol2.y.T[1:].tolist())
                            else:
                                t_vals.extend(sol2.t.tolist())
                                x_vals.extend(sol2.y.T.tolist())
                    break

            x = np.asarray(x_vals)
            t = np.asarray(t_vals)
            return x, t, True

        def __repr__(self):
            return f"SciPy solve_ivp integrator (method={self.method})"

# -----------------------
# Public model runner API
# -----------------------
def run_parcel_model(
    y0=None,
    aerosols=None,         # single dict/object OR list thereof
    V=1.0,
    T0=None, S0=None, P0=None,           # required if we need to build y0
    r_drys=None, Nis=None, kappas=None,  # optional manual arrays
    accom=ac_default,
    t_end=60.0,
    output_dt=1.0,
    solver_dt=None,
    terminate=False,
    terminate_depth=100.0,
    verbose=False,
    solver="auto",
    scipy_method="LSODA",
    **solver_args
):
    """
    Run the parcel model and return (x, z, meta).
    You may either:
      A) supply aerosols + (T0,S0,P0) and let this build y0,
      B) supply y0 and r_drys/Nis/kappas yourself,
      C) supply y0 + aerosols (we flatten to get r_drys/Nis/kappas; don’t rebuild y0).
    """
    names, lengths = [], []

    if (r_drys is not None) and (Nis is not None) and (kappas is not None):
        r_drys = np.asarray(r_drys, float).ravel()
        Nis    = np.asarray(Nis,    float).ravel()
        kappas = np.asarray(kappas, float).ravel()
        if y0 is None:
            raise ValueError("When passing r_drys/Nis/kappas directly, also pass y0.")
        if r_drys.size != Nis.size or r_drys.size != kappas.size:
            raise ValueError("Size mismatch among r_drys/Nis/kappas.")

    elif aerosols is not None:
        if y0 is None:
            if any(v is None for v in (T0, S0, P0)):
                raise ValueError("T0, S0, P0 must be given when building y0 from aerosols.")
            y0, r_drys, Nis, kappas, nr, names = build_initial_state(aerosols, V, T0, S0, P0)
            if isinstance(aerosols, (list, tuple)):
                lengths = [ (len(sp["r_drys"]) if isinstance(sp, dict) else len(getattr(sp, "r_drys")))
                           for sp in aerosols ]
            else:
                lengths = [nr]
        else:
            r_drys, Nis, kappas, lengths, names = flatten_aerosols(aerosols)

    else:
        raise ValueError("Provide either (aerosols …) or (y0 + r_drys/Nis/kappas).")

    if solver_dt is None:
        solver_dt = 10.0 * output_dt

    IntegratorType = Integrator.solver(solver)

    rhs = parcel_ode_sys  # supports callable V
    nr = int(np.asarray(r_drys).size)
    args = [nr, np.asarray(r_drys, float), np.asarray(Nis, float),
            V, np.asarray(kappas, float), float(accom)]

    y0 = np.asarray(y0, float)

    if IntegratorType.__name__.startswith("SciPy"):
        integ = IntegratorType(rhs, output_dt, solver_dt, y0, args,
                               console=bool(verbose), terminate=bool(terminate),
                               terminate_depth=float(terminate_depth),
                               method=scipy_method, **solver_args)
    else:
        integ = IntegratorType(rhs, output_dt, solver_dt, y0, args,
                               console=bool(verbose), terminate=bool(terminate),
                               terminate_depth=float(terminate_depth), **solver_args)

    try:
        x, t, success = integ.integrate(float(t_end))
    except ValueError as e:
        raise RuntimeError(f"Parcel integration failed: {e}")
    if not success:
        raise RuntimeError("Parcel integration did not complete successfully.")

    z = x[:, Z_IDX]
    meta = {"species_names": names, "bin_counts": lengths, "time": t}
    return x, z, meta


# -----------------------
# Optional quick example
# -----------------------
if __name__ == "__main__":
    # Build two sulfate modes as a smoke test
    sul1 = AerosolSpecies("sulfate",
                          Lognorm(mu=0.03, sigma=1.6, N=850.),  # microns
                          kappa=0.54, bins=120)
    sul2 = AerosolSpecies("sulfate-accum",
                          Lognorm(mu=0.09, sigma=1.6, N=50.),   # microns
                          kappa=0.54, bins=40)

    V = 1.0
    T0, S0, P0 = 274.0, -0.02, 77500.0
    y0, r_drys, Nis, kappas, nr, names = build_initial_state([sul1, sul2], V, T0, S0, P0)

    x, z, meta = run_parcel_model(
        y0=y0, r_drys=r_drys, Nis=Nis, kappas=kappas,
        V=V, t_end=250.0/V, output_dt=1.0, verbose=True, accom=0.3
    )
    print("Done. Samples:", x.shape[0], "Zmax (m):", z.max(), "nr:", nr)
