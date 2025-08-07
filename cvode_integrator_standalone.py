# -*- coding: utf-8 -*-
"""Standalone CVODEIntegrator interface for parcel model ODEs."""
import sys
import time
import warnings
import numpy as np
from abc import ABCMeta, abstractmethod

try:
    from assimulo.exception import TimeLimitExceeded
    from assimulo.problem import Explicit_Problem
    from assimulo.solvers.sundials import CVode, CVodeError
except ImportError:
    raise ImportError("Assimulo is required for CVODEIntegrator.")

# Compatibility timer
if sys.version_info[0] < 3:
    timer = time.clock
else:
    timer = time.process_time

# === Constants ===
STATE_VAR_MAP = {"z": 0, "T": 1, "S": 2}
N_STATE_VARS = 7
state_atol = [1e-4, 1e-4, 1e-4, 1e-10, 1e-10, 1e-4, 1e-8]
state_rtol = 1e-7


class Integrator(metaclass=ABCMeta):
    def __init__(self, rhs, output_dt, solver_dt, y0, args, t0=0.0, console=False):
        self.output_dt = output_dt
        self.solver_dt = solver_dt
        self.y0 = y0
        self.t0 = t0
        self.console = console
        self.args = args

        def _user_rhs(t, y):
            return rhs(y, t, *self.args)

        self.rhs = _user_rhs

    @abstractmethod
    def integrate(self, t_end, **kwargs):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class ExtendedProblem(Explicit_Problem):
    name = "Parcel model ODEs"
    sw0 = [True, False]
    t_cutoff = 1e5
    dS_dt = 1.0

    def __init__(self, rhs_fcn, rhs_args, terminate_depth, *args, **kwargs):
        self.rhs_fcn = rhs_fcn
        self.rhs_args = rhs_args
        self.V = rhs_args[3]  # updraft velocity
        self.terminate_time = terminate_depth / self.V
        super(Explicit_Problem, self).__init__(*args, **kwargs)

    def rhs(self, t, y, sw):
        if not sw[1]:
            dode_dt = self.rhs_fcn(t, y)
            self.dS_dt = dode_dt[N_STATE_VARS - 1]
        else:
            dode_dt = np.zeros(N_STATE_VARS + self.rhs_args[0])
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
    def __init__(
        self,
        rhs,
        output_dt,
        solver_dt,
        y0,
        args,
        t0=0.0,
        console=False,
        terminate=False,
        terminate_depth=100.0,
        **kwargs
    ):
        self.terminate = terminate
        super().__init__(rhs, output_dt, solver_dt, y0, args, t0, console)

        self.prob = (
            ExtendedProblem(self.rhs, self.args, terminate_depth, y0=self.y0)
            if terminate
            else Explicit_Problem(self.rhs, self.y0)
        )
        self.sim = self._setup_sim(**kwargs)

    def _setup_sim(self, **kwargs):
        sim = CVode(self.prob)
        sim.discr = "BDF"
        sim.maxord = 5

        sim.maxh = kwargs.get("maxh", min(0.1, self.output_dt))
        if "minh" in kwargs:
            sim.minh = kwargs["minh"]
        sim.iter = kwargs.get("iter", "Newton")
        if "linear_solver" in kwargs:
            sim.linear_solver = kwargs["linear_solver"]
        sim.maxsteps = kwargs.get("max_steps", 1000)
        sim.time_limit = kwargs.get("time_limit", 0.0)
        sim.report_continuously = sim.time_limit > 0
        sim.store_event_points = False

        nr = self.args[0]
        sim.rtol = state_rtol
        sim.atol = state_atol + [1e-12] * nr
        sim.verbosity = 40 if self.console else 50

        return sim

    def integrate(self, t_end, **kwargs):
        t_current = self.t0
        t_increment = self.solver_dt
        n_out = int(self.solver_dt / self.output_dt)

        txs, xxs = [], []
        n_steps = 1
        total_walltime = 0.0
        now = timer()

        while t_current < t_end:
            if self.console:
                delta_walltime = timer() - now
                total_walltime += delta_walltime
                state = self.y0 if n_steps == 1 else xxs[-1][-1]
                _z, _T, _S = state[0], state[1], state[2] * 100
                print(f"{n_steps:5d} {t_current:7.2f}s  {total_walltime:7.2f}s  {delta_walltime:8.2f}s | {_z:5.1f} {_T:7.2f} {_S:6.2f}%")

            try:
                now = timer()
                out_list = np.linspace(t_current, t_current + t_increment, n_out + 1)
                tx, xx = self.sim.simulate(t_current + t_increment, 0, out_list)
            except (CVodeError, TimeLimitExceeded) as e:
                raise RuntimeError("CVODE integration failed: %r" % e)

            if n_out == 1:
                txs.append(tx[-1])
                xxs.append(xx[-1])
            else:
                txs.extend(tx[:-1])
                xxs.append(xx[:-1])

            t_current = tx[-1]
            n_steps += 1

            if self.terminate and not self.sim.sw[0]:
                if self.console:
                    print("---- termination condition reached ----")
                break

        if self.console:
            print("---- end of integration loop ----")

        t = np.array(txs)
        x = np.array(xxs) if n_out == 1 else np.concatenate(xxs)
        return x, t, True

    def __repr__(self):
        return "CVODE integrator - direct Assimulo interface"
