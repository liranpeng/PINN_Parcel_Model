# eval_pinn.py
import argparse, json, os, math
import numpy as np
import torch

from ddp_pinn_helper import PINNParcelNet, prepare_sample_tensors, extract_meta, DTYPE

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - (ss_res / (ss_tot + 1e-12))

def safe_load_state(path, map_location="cpu"):
    sd = torch.load(path, map_location=map_location)
    # allow both plain state_dict and checkpoint dicts
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd

def build_from_cfg_or_cli(args):
    # Load JSON config if given
    cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = json.load(f)

    # CLI overrides (or standalone)
    if args.npz:        cfg["npz_path"] = args.npz
    if args.model:      cfg["model_out"] = args.model
    if args.hidden:     cfg["hidden"] = args.hidden
    if args.depth:      cfg["depth"] = args.depth
    if args.accom is not None: cfg["accom_val"] = args.accom
    if args.max_samples: cfg["max_samples"] = args.max_samples
    if args.save_preds:  cfg["save_preds"]  = args.save_preds

    # Defaults (match your training setup)
    cfg.setdefault("npz_path", "combined_outputs.npz")
    cfg.setdefault("model_out", "pinn_parcel.pt")
    cfg.setdefault("hidden", 256)
    cfg.setdefault("depth", 5)
    cfg.setdefault("accom_val", 0.3)
    cfg.setdefault("max_samples", 0)   # 0 => evaluate all
    cfg.setdefault("save_preds", None)

    return cfg

def main():
    ap = argparse.ArgumentParser(description="Evaluate trained PINN against parcel trajectories")
    ap.add_argument("--config", type=str, default=None, help="config.json (optional)")
    ap.add_argument("--npz", type=str, default=None, help="override dataset path")
    ap.add_argument("--model", type=str, default=None, help="override model weights path")
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--accom", type=float, default=None, help="accom_val used in training")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--max_samples", type=int, default=None, help="limit number of samples to evaluate")
    ap.add_argument("--save_preds", type=str, default=None, help="optional: write predictions to this .npz")
    args = ap.parse_args()

    cfg = build_from_cfg_or_cli(args)

    # Device
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    torch.set_grad_enabled(False)

    # Load data
    d = np.load(cfg["npz_path"], allow_pickle=True)
    x_list    = d["x_list"]
    y0_list   = d["y0_list"]
    meta_list = d["meta_list"]
    nsamples  = len(x_list)
    if cfg["max_samples"] and cfg["max_samples"] > 0:
        nsamples = min(nsamples, int(cfg["max_samples"]))

    # Infer n_bins and build model
    n_bins = x_list[0].shape[1] - 7
    model = PINNParcelNet(n_bins=n_bins,
                          hidden=cfg["hidden"],
                          depth=cfg["depth"],
                          cond_dim=8).to(device=device, dtype=DTYPE)
    sd = safe_load_state(cfg["model_out"], map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) or len(unexpected):
        print(f"[warn] load_state_dict: missing={missing}, unexpected={unexpected}")
    model.eval()

    # Accumulators
    abs_err_S = []; abs_err_T = []
    sq_err_S  = []; sq_err_T  = []
    all_S_true = []; all_S_pred = []
    all_T_true = []; all_T_pred = []

    # (optional) collect preds to save
    out_S, out_T, out_t = [], [], []

    for k in range(nsamples):
        x_np = x_list[k]; y0_np = y0_list[k]; meta = extract_meta(meta_list[k])
        ic, consts, t_nd_all, y_data, V_nd_all, n_bins_k, cond_vec = prepare_sample_tensors(
            x_np, y0_np, meta, cfg["accom_val"], device
        )
        assert n_bins_k == n_bins

        cond_inf = cond_vec.expand(t_nd_all.shape[0], -1)
        y_pred = model(t_nd_all.to(dtype=DTYPE),
                       V_nd_all.to(dtype=DTYPE),
                       cond_inf.to(dtype=DTYPE))

        # truth from x_np (physical units)
        S_true = torch.as_tensor(x_np[:, 6], dtype=DTYPE)
        T_true = torch.as_tensor(x_np[:, 2], dtype=DTYPE)

        S_hat = y_pred[:, 0].cpu()
        T_hat = y_pred[:, 1].cpu()

        # gather errors
        eS = (S_hat - S_true).abs().numpy()
        eT = (T_hat - T_true).abs().numpy()
        abs_err_S.append(eS); abs_err_T.append(eT)
        sq_err_S.append((S_hat - S_true).pow(2).numpy())
        sq_err_T.append((T_hat - T_true).pow(2).numpy())
        all_S_true.append(S_true.numpy()); all_S_pred.append(S_hat.numpy())
        all_T_true.append(T_true.numpy()); all_T_pred.append(T_hat.numpy())

        if cfg["save_preds"]:
            out_S.append(S_hat.numpy())
            out_T.append(T_hat.numpy())
            out_t.append(np.asarray(meta["time"]))

    # Flatten
    abs_err_S = np.concatenate(abs_err_S); abs_err_T = np.concatenate(abs_err_T)
    sq_err_S  = np.concatenate(sq_err_S);  sq_err_T  = np.concatenate(sq_err_T)
    S_true_f  = np.concatenate(all_S_true); S_pred_f = np.concatenate(all_S_pred)
    T_true_f  = np.concatenate(all_T_true); T_pred_f = np.concatenate(all_T_pred)

    # Metrics
    mae_S  = float(abs_err_S.mean()); rmse_S = float(np.sqrt(sq_err_S.mean())); r2_S = float(r2_score(S_true_f, S_pred_f))
    mae_T  = float(abs_err_T.mean()); rmse_T = float(np.sqrt(sq_err_T.mean())); r2_T = float(r2_score(T_true_f, T_pred_f))

    print("\n=== PINN evaluation vs. parcel (global over all time points) ===")
    print(f"S:  MAE={mae_S:.4e}   RMSE={rmse_S:.4e}   R^2={r2_S:.5f}")
    print(f"T:  MAE={mae_T:.4e}   RMSE={rmse_T:.4e}   R^2={r2_T:.5f}")
    print(f"Samples evaluated: {nsamples}")

    if cfg["save_preds"]:
        # ragged lists -> object arrays
        np.savez_compressed(
            cfg["save_preds"],
            S_hat_list=np.array(out_S, dtype=object),
            T_hat_list=np.array(out_T, dtype=object),
            t_list=np.array(out_t, dtype=object),
        )
        print(f"Saved predictions to {cfg['save_preds']}")

if __name__ == "__main__":
    main()

