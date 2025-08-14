# ddp_pinn_helper.py
import os, warnings, time
import datetime as dt
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd import grad
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ============================================================
# Global config (float32 for DDP speed & consistency)
# ============================================================
DTYPE = torch.float32

# ===============================
# Nondimensional / scaling to physical
# ===============================
T_ref  = 273.15
dT_ref = 30.0
P_ref  = 8.0e4
S_ref  = 0.01
wv_ref = 0.02
wc_ref = 1e-3
r_ref  = 1e-6

# ===============================
# Thermo / microphysics constants
# ===============================
PI    = np.pi
g     = 9.81
Cp    = 1004.0
L     = 2.25e6
rho_w = 1e3
R     = 8.314
Mw    = 18.0 / 1e3   # kg/mol
Ma    = 28.9 / 1e3   # kg/mol
Rd    = R / Ma

# ===============================
# Utility
# ===============================
def to_t(x):
    return torch.as_tensor(x, dtype=DTYPE)

def sigma_w(T: torch.Tensor) -> torch.Tensor:
    return 0.0761 - 1.55e-4 * (T - 273.15)

def es(Tc: torch.Tensor) -> torch.Tensor:
    # saturation vapor pressure (Pa) with Tc in °C
    return 611.2 * torch.exp(17.67 * Tc / (Tc + 243.5))

def dv_eff(T, r, P, accom):
    P_atm = P * 1.01325e-5
    dv_cont = 1e-4 * (0.211 / P_atm) * (T / 273.0)**1.94
    denom = 1.0 + (dv_cont / (accom * torch.clamp(r, min=1e-30))) * torch.sqrt((2.0 * np.pi * Mw) / (R * T))
    return dv_cont / denom

def ka_eff(T, r, rho):
    ka_cont = 1e-3 * (4.39 + 0.071 * T)
    denom = 1.0 + (ka_cont / (0.96 * torch.clamp(r, min=1e-30) * rho * Cp)) * torch.sqrt((2.0 * np.pi * Ma) / (R * T))
    return ka_cont / denom

def rho_air_dry(P, T):
    return P / (Rd * T)

# ===============================
# PINN model
# ===============================
class ResidualBlock(nn.Module):
    def __init__(self, dim:int):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.Tanh()
        nn.init.xavier_uniform_(self.lin1.weight); nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight); nn.init.zeros_(self.lin2.bias)
    def forward(self, x):
        h = self.act(self.lin1(x))
        h = self.lin2(h)
        return self.act(x + h)

class PINNParcelNet(nn.Module):
    """
    Inputs:
      t_nd : (N,1) normalized time in [0,1]
      V_nd : (N,1) normalized updraft
      cond : (N,C) conditioning: normalized ICs + aerosol stats
    Outputs (physical units):
      [S, T, P, wv, wc, r_1..r_N]
    """
    def __init__(self, n_bins:int, hidden:int=256, depth:int=5, cond_dim:int=8):
        super().__init__()
        self.n_bins = n_bins
        self.cond_dim = cond_dim
        in_dim = 2 + cond_dim  # [t_nd, V_nd, cond...]

        trunk = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth-1):
            trunk += [ResidualBlock(hidden)]
        self.trunk = nn.Sequential(*trunk)

        self.head_bulk = nn.Linear(hidden, 5)          # S*, T*, P*, wv*, wc*
        self.head_r    = nn.Linear(hidden, n_bins)     # r*

        nn.init.xavier_uniform_(self.head_bulk.weight); nn.init.zeros_(self.head_bulk.bias)
        nn.init.xavier_uniform_(self.head_r.weight);    nn.init.zeros_(self.head_r.bias)

    def forward(self, t_nd:torch.Tensor, V_nd:torch.Tensor, cond:Optional[torch.Tensor]) -> torch.Tensor:
        if cond is None:
            raise ValueError("cond vector must be provided")
        if cond.shape[0] == 1:
            cond = cond.expand(t_nd.shape[0], -1)
        x  = torch.cat([t_nd, V_nd, cond], dim=1)

        h = self.trunk[0](x)
        h = self.trunk[1](h)
        for blk in self.trunk[2:]:
            h = blk(h)

        y_bulk = self.head_bulk(h)  # (N,5)
        y_r    = self.head_r(h)     # (N,n_bins)

        # map to physical units (physics will use these)
        S  = torch.tanh(y_bulk[:, 0:1]) * S_ref
        T  = T_ref + dT_ref * y_bulk[:, 1:2]
        P  = P_ref + (dT_ref * 1000.0) * y_bulk[:, 2:3]
        wv = wv_ref * torch.tanh(y_bulk[:, 3:4])
        wc = wc_ref * torch.tanh(y_bulk[:, 4:5])
        r  = torch.nn.functional.softplus(y_r) * r_ref + 1e-12
        return torch.cat([S, T, P, wv, wc, r], dim=1)

# ===============================
# Data helpers
# ===============================
def extract_meta(m):
    return m if isinstance(m, dict) else m.item()

def infer_V_from_z_t(x_np: np.ndarray, t_np: np.ndarray) -> float:
    z = x_np[:, 0]
    t0, t1 = float(t_np[0]), float(t_np[-1])
    if t1 <= t0:
        return 1.0
    return float((z[-1] - z[0]) / (t1 - t0))

def _get_vec_from_meta(meta: Dict[str, Any], key: str, n_bins: int):
    if key not in meta or meta[key] is None:
        return None
    arr = np.asarray(meta[key], dtype=np.float64).ravel()
    if arr.size != n_bins or not np.all(np.isfinite(arr)):
        return None
    return arr

def build_cond_vector_nd(ic: Dict[str, torch.Tensor],
                         consts: Dict[str, torch.Tensor],
                         device: torch.device) -> torch.Tensor:
    """
    Normalized conditioning vector (shape: (1,8)), ~O(1) scales:
      [S0/S_ref, (T0-T_ref)/dT_ref, (P0-P_ref)/(dT_ref*1000),
       wv0/wv_ref, wc0/wc_ref, log10(sum(Ni)), mean(r_dry)/1e-6, mean(kappa)]
    """
    S0n  = ic["S0"]  / to_t(S_ref)
    T0n  = (ic["T0"] - to_t(T_ref)) / to_t(dT_ref)
    P0n  = (ic["P0"] - to_t(P_ref)) / to_t(dT_ref * 1000.0)
    wv0n = ic["wv0"] / to_t(wv_ref)
    wc0n = ic["wc0"] / to_t(wc_ref)

    Nis = consts["Nis"].to(device=device, dtype=DTYPE)
    rds = consts["r_drys"].to(device=device, dtype=DTYPE)
    kap = consts["kappas"].to(device=device, dtype=DTYPE)

    sumNi = torch.clamp(torch.sum(Nis), min=1e-30)
    logNi = torch.log10(sumNi)
    mean_rd_um = torch.mean(rds) / to_t(1e-6)
    mean_k = torch.mean(kap)

    vec = torch.cat([S0n, T0n, P0n, wv0n, wc0n,
                     logNi.view(1,1), mean_rd_um.view(1,1), mean_k.view(1,1)], dim=1)
    return vec.to(dtype=DTYPE, device=device)

def prepare_sample_tensors(x_np: np.ndarray, y0_np: np.ndarray, meta: Dict[str,Any],
                           accom_val: float, device: torch.device):
    n_bins = x_np.shape[1] - 7
    t_np = meta["time"]

    # ICs (physical)
    S0, T0, P0 = y0_np[6:7], y0_np[2:3], y0_np[1:2]
    wv0, wc0   = y0_np[3:4], y0_np[4:5]
    r0s        = y0_np[7:7+n_bins]
    ic = {
        "S0":  torch.tensor(S0,  device=device, dtype=DTYPE).view(1,1),
        "T0":  torch.tensor(T0,  device=device, dtype=DTYPE).view(1,1),
        "P0":  torch.tensor(P0,  device=device, dtype=DTYPE).view(1,1),
        "wv0": torch.tensor(wv0, device=device, dtype=DTYPE).view(1,1),
        "wc0": torch.tensor(wc0, device=device, dtype=DTYPE).view(1,1),
        "r0s": torch.tensor(r0s, device=device, dtype=DTYPE).view(1,-1),
    }

    # Aerosols (physical, per-bin)
    r_drys_np = _get_vec_from_meta(meta, "r_drys", n_bins)
    Nis_np    = _get_vec_from_meta(meta, "Nis",    n_bins)
    kappas_np = _get_vec_from_meta(meta, "kappas", n_bins)
    if (r_drys_np is None) or (Nis_np is None) or (kappas_np is None):
        warnings.warn("r_drys/Nis/kappas missing or wrong length; using placeholders.")
        if r_drys_np is None: r_drys_np = np.ones(n_bins) * 0.03e-6
        if Nis_np    is None: Nis_np    = np.ones(n_bins) * 1.0e8
        if kappas_np is None: kappas_np = np.full(n_bins, 0.54)

    # time / updraft normalization
    V_val = infer_V_from_z_t(x_np, t_np)
    t_all = torch.tensor(t_np, device=device, dtype=DTYPE).view(-1,1)
    t_ref = float(t_np[-1] - t_np[0]) if (t_np[-1] - t_np[0]) > 0 else 1.0
    t_nd_all = (t_all - t_all[0]) / to_t(t_ref)

    V_scale = max(1.0, abs(V_val))
    V_nd_all = torch.full_like(t_nd_all, fill_value=(V_val / V_scale), dtype=DTYPE, device=device)

    # supervised targets (physical)
    y_data = torch.tensor(
        np.column_stack([
            x_np[:,6],     # S
            x_np[:,2],     # T
            x_np[:,1],     # P
            x_np[:,3],     # wv
            x_np[:,4],     # wc
            x_np[:,7:7+n_bins]
        ]),
        device=device, dtype=DTYPE
    )

    consts = {
        "r_drys": torch.tensor(r_drys_np, device=device, dtype=DTYPE),
        "Nis":    torch.tensor(Nis_np,    device=device, dtype=DTYPE),
        "kappas": torch.tensor(kappas_np, device=device, dtype=DTYPE),
        "accom":  torch.tensor(float(accom_val), device=device, dtype=DTYPE),
        "V_phys": torch.tensor(float(V_val),     device=device, dtype=DTYPE),
        "t_ref":  torch.tensor(float(t_ref),     device=device, dtype=DTYPE),
        "V_scale":torch.tensor(float(V_scale),   device=device, dtype=DTYPE),
    }

    # conditioning vector (normalized)
    cond_vec = build_cond_vector_nd(ic, consts, device)  # (1,8)

    return ic, consts, t_nd_all, y_data, V_nd_all, n_bins, cond_vec

# ===============================
# Loss (Physics + IC + Data)
# ===============================
def pinn_losses(model: nn.Module,
                t_nd: torch.Tensor,
                V_nd: torch.Tensor,
                t_ref: float,
                consts: Dict[str,torch.Tensor],
                ic: Dict[str,torch.Tensor],
                W_PHYS: float,
                W_IC: float,
                W_DATA: float,
                cond_vec: Optional[torch.Tensor] = None,
                data_t_nd: Optional[torch.Tensor] = None,
                data_y: Optional[torch.Tensor] = None,
                ic_auto_balance: bool = True) -> Dict[str, torch.Tensor]:

    cond = cond_vec.expand(t_nd.shape[0], -1) if cond_vec is not None else None

    # forward (physical outputs)
    y_hat = model(t_nd, V_nd, cond)  # (N, 5+Nbins)
    S  = y_hat[:, 0:1]
    T  = y_hat[:, 1:2]
    P  = y_hat[:, 2:3]
    wv = y_hat[:, 3:4]
    wc = y_hat[:, 4:5]
    r  = y_hat[:, 5:]
    Nc, Nbins = r.shape

    # d/dt via chain rule (t_nd -> t)
    scale_dt = 1.0 / t_ref
    d_cols = []
    for k in range(y_hat.shape[1]):
        gk_nd = grad(y_hat[:, k].sum(), t_nd, create_graph=True, retain_graph=True)[0]
        d_cols.append(gk_nd * scale_dt)
    dY_dt = torch.cat(d_cols, dim=1)
    dS_dt   = dY_dt[:, 0:1]
    dT_dt   = dY_dt[:, 1:2]
    dP_dt   = dY_dt[:, 2:3]
    dwv_dt  = dY_dt[:, 3:4]
    dwc_dt  = dY_dt[:, 4:5]
    dr_dt_full = dY_dt[:, 5:]

    # Broadcast sample constants
    accom  = consts["accom"]
    r_drys = consts["r_drys"].view(1, Nbins).repeat(Nc, 1)
    Nis    = consts["Nis"].view(1, Nbins).repeat(Nc, 1)
    kappas = consts["kappas"].view(1, Nbins).repeat(Nc, 1)
    V_phys = consts["V_phys"].view(1,1).repeat(Nc,1)

    # saturation & density (physical)
    pv_sat = torch.clamp(es(T - 273.15), 50.0, 1.5e5)
    rho_dry = torch.clamp(rho_air_dry(P, T), 0.1, 5.0)

    # microphysics (growth coefficient)
    dv_r = dv_eff(T, r, P, accom)
    ka_r = ka_eff(T, r, rho_dry)
    G_a = (rho_w * R * T) / (pv_sat * dv_r * Mw)
    G_b = (L * rho_w * ((L * Mw / (R * T)) - 1.0)) / (ka_r * T)
    G = 1.0 / (G_a + G_b)

    # Köhler-equilibrium Seq
    r_safe = torch.clamp(r, min=1e-30)
    r3, rd3 = r_safe**3.0, r_drys**3.0
    den = r3 - rd3 * (1.0 - kappas)
    den = torch.where(den <= 0, torch.full_like(den, 1e-30), den)
    A = (2.0 * Mw * sigma_w(T)) / (R * T * rho_w * r_safe)
    A = torch.clamp(A, -100.0, 100.0)
    B_when_pos  = (r3 - rd3) / den
    B_when_zero = torch.ones_like(r_safe)
    B = torch.where(kappas > 0.0, B_when_pos, B_when_zero)
    Seq = torch.exp(A) * B - 1.0

    dr_dt_phys = (G / r_safe) * (S - Seq)

    # mass / thermo / dynamics
    sum_term = torch.sum(Nis * (r_safe**2.0) * dr_dt_full, dim=1, keepdim=True)
    dwc_dt_phys = 4.0 * np.pi * rho_w * sum_term / torch.clamp(rho_dry, min=1e-30)
    dwv_dt_phys = -dwc_dt_phys
    dT_dt_phys = -g * V_phys / Cp - (L * dwv_dt_phys / Cp)
    dP_dt_phys = -rho_dry * g * V_phys

    alpha = (g * Mw * L) / (Cp * R * (T**2.0)) - (g * Ma) / (R * T)
    gamma = (P * Ma) / (Mw * pv_sat) + (Mw * L * L) / (Cp * R * T**2.0)
    dS_dt_phys = alpha * V_phys - gamma * dwc_dt_phys

    # Auto-balancing weights for physics residuals
    with torch.no_grad():
        def med(x): return torch.median(torch.abs(x)) + 1e-12
        w_r  = 1.0 / (med(dr_dt_phys) * med(dr_dt_full))
        w_T  = 1.0 / (med(dT_dt_phys) * med(dT_dt))
        w_P  = 1.0 / (med(dP_dt_phys) * med(dP_dt))
        w_S  = 1.0 / (med(dS_dt_phys) * med(dS_dt))
        w_wc = 1.0 / (med(dwc_dt_phys) * med(dwc_dt))
        w_wv = 1.0 / (med(dwv_dt_phys) * med(dwv_dt))

    loss_phys = (
        w_r  * torch.mean((dr_dt_full - dr_dt_phys)**2) +
        w_T  * torch.mean((dT_dt      - dT_dt_phys )**2) +
        w_P  * torch.mean((dP_dt      - dP_dt_phys )**2) +
        w_S  * torch.mean((dS_dt      - dS_dt_phys )**2) +
        (w_wc * torch.mean((dwc_dt - dwc_dt_phys)**2) +
         w_wv * torch.mean((dwv_dt - dwv_dt_phys)**2))
    )

    # IC consistency at t*=0
    t0_nd = torch.zeros_like(t_nd[:1])
    cond0 = cond_vec.expand(1, -1) if cond_vec is not None else None
    y0_hat = model(t0_nd, V_nd[:1], cond0)
    S0h, T0h, P0h = y0_hat[:,0:1], y0_hat[:,1:2], y0_hat[:,2:3]
    wv0h, wc0h    = y0_hat[:,3:4], y0_hat[:,4:5]
    r0h           = y0_hat[:,5:]

    if ic_auto_balance:
        with torch.no_grad():
            def med(x): return torch.median(torch.abs(x)) + 1e-12
            wS = 1.0 / med(ic["S0"])
            wT = 1.0 / med(ic["T0"])
            wP = 1.0 / med(ic["P0"])
            wv = 1.0 / med(ic["wv0"])
            wc = 1.0 / med(ic["wc0"])
            wr = 1.0 / med(ic["r0s"])
        loss_ic = (
            wS*torch.mean((S0h - ic["S0"])**2) +
            wT*torch.mean((T0h - ic["T0"])**2) +
            wP*torch.mean((P0h - ic["P0"])**2) +
            wv*torch.mean((wv0h - ic["wv0"])**2) +
            wc*torch.mean((wc0h - ic["wc0"])**2) +
            wr*torch.mean((r0h  - ic["r0s"])**2)
        )
    else:
        loss_ic = (torch.mean((S0h - ic["S0"])**2) +
                   torch.mean((T0h - ic["T0"])**2) +
                   torch.mean((P0h - ic["P0"])**2) +
                   torch.mean((wv0h - ic["wv0"])**2) +
                   torch.mean((wc0h - ic["wc0"])**2) +
                   torch.mean((r0h  - ic["r0s"])**2))

    # optional supervised loss (physical)
    loss_data = torch.tensor(0.0, dtype=DTYPE, device=y_hat.device)
    if (data_t_nd is not None) and (data_y is not None):
        cond_data = cond_vec.expand(data_t_nd.shape[0], -1) if cond_vec is not None else None
        y_sup = model(data_t_nd, V_nd[:data_t_nd.shape[0]], cond_data)
        loss_data = torch.mean((y_sup - data_y)**2)

    total = float(W_PHYS)*loss_phys + float(W_IC)*loss_ic + float(W_DATA)*loss_data
    return {"total": total, "phys": loss_phys, "ic": loss_ic, "data": loss_data}

# ===============================
# DDP init/cleanup helpers (env://)
# ===============================
def _get_device_from_env() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
        except Exception:
            pass
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")

def _init_ddp() -> Tuple[int,int]:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=dt.timedelta(seconds=90)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def _cleanup_ddp():
    if dist.is_initialized():
        try:
            dist.barrier(timeout=dt.timedelta(seconds=10))
        except Exception:
            pass
        dist.destroy_process_group()

# ===============================
# Epoch index builder (DDP-safe: equal steps/rank)
# ===============================
def _equal_split(indices: np.ndarray, world_size: int, rank: int) -> np.ndarray:
    """
    Split 'indices' into equal-size contiguous chunks across ranks.
    Drops remainder to keep identical iteration counts per rank.
    If len(indices) < world_size, repeat with replacement so every rank has work.
    """
    n = len(indices)
    if n == 0:
        return np.array([0], dtype=int)
    per_rank = n // world_size
    if per_rank == 0:
        expanded = np.resize(indices, world_size)  # repeat as needed
        return np.array([expanded[rank]], dtype=int)
    n_eff = per_rank * world_size
    indices = indices[:n_eff]
    chunks = np.array_split(indices, world_size)
    return chunks[rank]

def _make_epoch_indices(nsamples: int,
                        world_size: int,
                        rank: int,
                        rng: np.random.Generator,
                        samples_per_epoch: Optional[int],
                        reshuffle_across_ranks: bool) -> np.ndarray:
    """
    Return the indices THIS RANK should process for the current epoch.
    """
    if (samples_per_epoch is None) or (samples_per_epoch <= 0) or (samples_per_epoch >= nsamples):
        base = np.arange(nsamples)
        if reshuffle_across_ranks:
            base = rng.permutation(base)
        return _equal_split(base, world_size, rank)

    K = int(min(samples_per_epoch, nsamples))
    base = rng.permutation(nsamples) if reshuffle_across_ranks else np.arange(nsamples)
    chosen = base[:K]
    return _equal_split(chosen, world_size, rank)

# ===============================
# EMA helpers
# ===============================
def _init_ema(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.module.named_parameters()}

def _update_ema(ema: Dict[str, torch.Tensor], model: nn.Module, decay: float):
    with torch.no_grad():
        for n, p in model.module.named_parameters():
            ema[n].mul_(decay).add_(p.detach(), alpha=1.0 - decay)

def _state_dict_from_ema(ema: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # move tensors to CPU for saving
    return {n: t.detach().cpu() for n, t in ema.items()}

# ===============================
# Public entry from train_ddp.py
# ===============================
def ddp_train_entry(cfg: Dict[str, Any]):
    """
    Called by each torchrun rank.
    Uses env:// rendezvous, sets device from LOCAL_RANK,
    trains, saves (rank0), and shuts down cleanly.

    cfg keys this function uses:
      npz_path, hidden, depth, lr, epochs, n_colloc,
      W_PHYS, W_IC, W_DATA, accom_val,
      samples_per_epoch, reshuffle_across_ranks,
      print_every, model_out, preds_out,
      ckpt_dir, save_every, resume_from, run_inference,
      sup_points, edge_supervision_frac, ic_auto_balance,
      use_scheduler, lr_factor, lr_patience, min_lr,
      clip_grad_norm, ema_decay, use_ema_for_eval
    """
    device = _get_device_from_env()
    rank, world_size = _init_ddp()

    if cfg.get("detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)

    try:
        # -------- Load data --------
        d = np.load(cfg["npz_path"], allow_pickle=True)
        x_list    = d["x_list"]
        y0_list   = d["y0_list"]
        meta_list = d["meta_list"]

        nsamples = len(x_list)
        assert nsamples > 0, "Empty dataset."
        n_bins0 = x_list[0].shape[1] - 7
        for k in range(nsamples):
            assert (x_list[k].shape[1] - 7) == n_bins0, "All samples must share same bin count."

        # -------- Model / Opt --------
        cond_dim = 8
        model = PINNParcelNet(n_bins=n_bins0,
                              hidden=int(cfg.get("hidden", 256)),
                              depth=int(cfg.get("depth", 5)),
                              cond_dim=cond_dim).to(device=device, dtype=DTYPE)

        model = DDP(model,
                    device_ids=[device.index] if device.type == "cuda" else None,
                    broadcast_buffers=False,
                    find_unused_parameters=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 3e-4)))

        # optional scheduler
        use_scheduler   = bool(cfg.get("use_scheduler", 1))
        lr_factor       = float(cfg.get("lr_factor", 0.5))
        lr_patience     = int(cfg.get("lr_patience", 6))
        min_lr          = float(cfg.get("min_lr", 1e-7))
        scheduler = None
        if use_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, factor=lr_factor, patience=lr_patience,
                                          min_lr=min_lr, verbose=(rank==0))

        # loss weights
        W_PHYS = float(cfg.get("W_PHYS", 1.0))
        W_IC   = float(cfg.get("W_IC",   1.0))
        W_DATA = float(cfg.get("W_DATA", 0.1))
        ic_auto_balance = bool(cfg.get("ic_auto_balance", True))

        # EMA
        ema_decay = float(cfg.get("ema_decay", 0.999))
        use_ema_for_eval = bool(cfg.get("use_ema_for_eval", 1))
        ema = _init_ema(model)

        # ----- Resume (weights only) -----
        resume_path = cfg.get("resume_from", None)
        if not resume_path:
            # auto-detect last.pt if present
            ckpt_dir_default = cfg.get("ckpt_dir", "checkpoints")
            candidate = os.path.join(ckpt_dir_default, "last.pt")
            if os.path.isfile(candidate):
                resume_path = candidate
        if resume_path and os.path.isfile(resume_path):
            map_location = device if device.type == "cpu" else {"cuda:0": f"cuda:{device.index}"}
            try:
                state = torch.load(resume_path, map_location=map_location)
                missing, unexpected = model.module.load_state_dict(state, strict=False)
                if rank == 0:
                    print(f"[rank0] Resumed from {resume_path} "
                          f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
                # sync EMA to current model on resume
                ema = _init_ema(model)
            except Exception as e:
                if rank == 0:
                    print(f"[rank0] WARNING: failed to resume from {resume_path}: {e}", flush=True)

        # -------- Train --------
        samples_per_epoch_cfg = int(cfg.get("samples_per_epoch", 0)) or None
        reshuffle = bool(cfg.get("reshuffle_across_ranks", True))
        print_every = int(cfg.get("print_every", 10))
        ckpt_dir = cfg.get("ckpt_dir", "checkpoints")
        save_every = int(cfg.get("save_every", 0))  # 0 => skip named epoch saves, but still write last.pt
        clip_val = float(cfg.get("clip_grad_norm", 0.5))  # tighter than 1.0
        sup_points = int(cfg.get("sup_points", 64))
        edge_frac = float(cfg.get("edge_supervision_frac", 0.25))

        os.makedirs(ckpt_dir, exist_ok=True)

        total_epochs = int(cfg.get("epochs", 100))
        for ep in range(1, total_epochs + 1):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            rng = np.random.default_rng(seed=ep + rank*17)
            order = _make_epoch_indices(
                nsamples=nsamples,
                world_size=world_size,
                rank=rank,
                rng=rng,
                samples_per_epoch=samples_per_epoch_cfg,
                reshuffle_across_ranks=reshuffle
            )
            rng.shuffle(order)  # local shuffle

            epoch_tot = epoch_phys = epoch_ic = epoch_data = 0.0
            seen = 0

            for k in order:
                x_np = x_list[k]; y0_np = y0_list[k]
                meta = extract_meta(meta_list[k])
                ic, consts, t_nd_all, y_data, V_nd_all, n_bins, cond_vec = prepare_sample_tensors(
                    x_np, y0_np, meta, float(cfg.get("accom_val", 0.3)), device
                )

                # collocation points (requires_grad for autograd)
                t_nd = torch.rand((int(cfg.get("n_colloc", 64)),1), device=device, dtype=DTYPE, requires_grad=True)
                V_nd = torch.full_like(t_nd, fill_value=V_nd_all[0].item(), dtype=DTYPE, device=device)

                # supervised subsample (physical) with edge bias
                n_all = t_nd_all.shape[0]
                sp = min(sup_points, n_all)
                n_edge = min(int(sp * edge_frac), n_all // 2)
                if n_edge > 0:
                    left = torch.arange(n_edge, device=device)
                    right = torch.arange(n_all - n_edge, n_all, device=device)
                    mask = torch.ones(n_all, dtype=torch.bool, device=device)
                    mask[left] = False; mask[right] = False
                    rest_candidates = torch.arange(n_all, device=device)[mask]
                    if rest_candidates.numel() > 0 and (sp - 2*n_edge) > 0:
                        rest = rest_candidates[torch.randperm(rest_candidates.numel(), device=device)[:(sp - 2*n_edge)]]
                        idx = torch.cat([left, right, rest])
                    else:
                        idx = torch.cat([left, right])
                else:
                    idx = torch.randperm(n_all, device=device)[:sp]

                data_t_nd = t_nd_all[idx].detach().to(dtype=DTYPE)
                data_y    = y_data[idx].detach().to(dtype=DTYPE)

                t_ref = float(consts["t_ref"].item())

                # sanity
                assert torch.isfinite(t_nd).all() and torch.isfinite(V_nd).all()
                assert torch.isfinite(data_t_nd).all() and torch.isfinite(data_y).all()

                model.train()
                optimizer.zero_grad(set_to_none=True)
                losses = pinn_losses(model, t_nd, V_nd, t_ref, consts, ic,
                                     W_PHYS, W_IC, W_DATA,
                                     cond_vec=cond_vec,
                                     data_t_nd=data_t_nd, data_y=data_y,
                                     ic_auto_balance=ic_auto_balance)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                optimizer.step()

                # EMA update
                _update_ema(ema, model, ema_decay)

                epoch_tot  += float(losses["total"].item())
                epoch_phys += float(losses["phys"].item())
                epoch_ic   += float(losses["ic"].item())
                epoch_data += float(losses["data"].item())
                seen += 1

            # average metrics across ranks; include actual 'seen' count
            metrics = torch.tensor([epoch_tot, epoch_phys, epoch_ic, epoch_data, float(seen)],
                                   dtype=DTYPE, device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            epoch_sec = torch.tensor([t1 - t0], dtype=DTYPE, device=device)
            dist.all_reduce(epoch_sec, op=dist.ReduceOp.MAX)   # slowest rank
            secs_per_epoch = float(epoch_sec.item())

            # compute global averages for logging & scheduler
            tot, phys, icc, data, count = metrics.tolist()
            m = max(count, 1.0)  # total samples processed globally this epoch
            avg_total = tot / m

            # scheduler step (same value on all ranks)
            if scheduler is not None:
                scheduler.step(avg_total)

            if rank == 0 and ((ep % print_every == 0) or ep == 1):
                eta_hours = (total_epochs - ep) * secs_per_epoch / 3600.0
                print(f"[epoch {ep:4d}/{total_epochs}] "
                      f"total={tot/m:.3e} phys={phys/m:.3e} ic={icc/m:.3e} data={data/m:.3e} | "
                      f"samples/epoch={int(count)} | time/epoch={secs_per_epoch:.2f}s  ETA~{eta_hours:.2f}h",
                      flush=True)

            # --- Save checkpoints at end of epoch (rank 0) ---
            try:
                dist.barrier()
            except Exception:
                pass

            if rank == 0:
                # periodic named checkpoint (raw weights)
                if save_every > 0 and (ep % save_every == 0):
                    ckpt_path = os.path.join(ckpt_dir, f"pinn_epoch_{ep:04d}.pt")
                    torch.save(model.module.state_dict(), ckpt_path)
                    # also store EMA snapshot
                    ckpt_ema_path = os.path.join(ckpt_dir, f"pinn_epoch_{ep:04d}_ema.pt")
                    torch.save(_state_dict_from_ema(ema), ckpt_ema_path)

                # rolling last.pt (raw) and last_ema.pt (EMA)
                last_path = os.path.join(ckpt_dir, "last.pt")
                tmp_path = last_path + ".tmp"
                torch.save(model.module.state_dict(), tmp_path)
                os.replace(tmp_path, last_path)

                last_ema_path = os.path.join(ckpt_dir, "last_ema.pt")
                tmp_ema = last_ema_path + ".tmp"
                torch.save(_state_dict_from_ema(ema), tmp_ema)
                os.replace(tmp_ema, last_ema_path)

        # -------- Save final + optional inference on rank0 --------
        try:
            dist.barrier()
        except Exception:
            pass

        if rank == 0:
            # choose which weights to export as "model_out"
            if use_ema_for_eval:
                torch.save(_state_dict_from_ema(ema), cfg["model_out"])
            else:
                torch.save(model.module.state_dict(), cfg["model_out"])
            print(f"[rank0] saved model to {cfg['model_out']}", flush=True)

        run_inf = int(cfg.get("run_inference", 1))
        if rank == 0 and run_inf:
            print("[rank0] running inference on all samples…", flush=True)
            S_all, T_all, P_all, wv_all, wc_all, r_all, t_all, V_list = [], [], [], [], [], [], [], []
            model.eval()

            # if using EMA for eval, create a shadow copy of params for inference
            if use_ema_for_eval:
                with torch.no_grad():
                    for n, p in model.module.named_parameters():
                        p.data.copy_(ema[n])

            with torch.no_grad():
                for k in range(nsamples):
                    x_np = x_list[k]; y0_np = y0_list[k]; meta = extract_meta(meta_list[k])
                    ic, consts, t_nd_all, _, V_nd_all, n_bins, cond_vec = prepare_sample_tensors(
                        x_np, y0_np, meta, float(cfg.get("accom_val", 0.3)), device
                    )
                    cond_inf = cond_vec.expand(t_nd_all.shape[0], -1)
                    y_pred = model(t_nd_all.to(dtype=DTYPE), V_nd_all.to(dtype=DTYPE), cond_inf).cpu().numpy()
                    S_all.append(y_pred[:,0]); T_all.append(y_pred[:,1]); P_all.append(y_pred[:,2])
                    wv_all.append(y_pred[:,3]); wc_all.append(y_pred[:,4]); r_all.append(y_pred[:,5:])
                    t_all.append(meta["time"])
                    V_list.append(float(consts["V_phys"].item()))

            np.savez_compressed(
                cfg["preds_out"],
                S_list=np.array(S_all, dtype=object),
                T_list=np.array(T_all, dtype=object),
                P_list=np.array(P_all, dtype=object),
                wv_list=np.array(wv_all, dtype=object),
                wc_list=np.array(wc_all, dtype=object),
                r_list=np.array(r_all, dtype=object),
                t_list=np.array(t_all, dtype=object),
                V_list=np.array(V_list, dtype=float),
            )
            print(f"[rank0] wrote predictions to {cfg['preds_out']}", flush=True)

    finally:
        _cleanup_ddp()

