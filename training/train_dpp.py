# train_ddp.py
import argparse
import json
import os
import sys

import ddp_pinn_helper as helper


def parse_args():
    ap = argparse.ArgumentParser(description="DDP trainer launcher for PINN parcel model")

    # ---------- Primary: load from JSON config ----------
    ap.add_argument("--config", type=str, default=None,
                    help="Path to JSON config. Fields can be overridden by CLI flags below.")

    # ---------- Dataset / training length ----------
    ap.add_argument("--npz", type=str, default=None,
                    help="Path to combined_outputs.npz (overrides config npz_path)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Number of epochs to run in THIS job")
    ap.add_argument("--samples_per_epoch", type=int, default=None,
                    help="Global number of trajectories per epoch; 0/None or >= dataset size => full dataset")
    ap.add_argument("--reshuffle_across_ranks", type=int, default=None,
                    help="1 to reshuffle the global pool each epoch, 0 to keep fixed order")

    # ---------- Model / optimizer / collocation ----------
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n_colloc", type=int, default=None)

    # ---------- Supervised slice controls ----------
    ap.add_argument("--sup_points", type=int, default=None,
                    help="Supervised time steps per sample (default from config)")
    ap.add_argument("--edge_supervision_frac", type=float, default=None,
                    help="Fraction of sup_points pulled from start/end of each profile (0..1)")

    # ---------- Physics weights / constants ----------
    ap.add_argument("--accom", type=float, default=None,
                    help="Accommodation coefficient (accom_val)")
    ap.add_argument("--w_phys", type=float, default=None)
    ap.add_argument("--w_ic", type=float, default=None)
    ap.add_argument("--w_data", type=float, default=None)
    ap.add_argument("--ic_auto_balance", type=int, default=None,
                    help="1 to auto-balance IC terms, 0 to use raw weights")

    # ---------- Scheduler / regularization / EMA ----------
    ap.add_argument("--use_scheduler", type=int, default=None,
                    help="1 to enable ReduceLROnPlateau, 0 to disable")
    ap.add_argument("--lr_factor", type=float, default=None,
                    help="LR reduce factor on plateau (e.g., 0.5)")
    ap.add_argument("--lr_patience", type=int, default=None,
                    help="Epochs with no improvement before LR is reduced")
    ap.add_argument("--min_lr", type=float, default=None,
                    help="Lower bound on LR when scheduler is enabled")

    ap.add_argument("--clip_grad_norm", type=float, default=None,
                    help="Max global gradient norm; <=0 disables clipping")

    ap.add_argument("--ema_decay", type=float, default=None,
                    help="EMA decay for model weights (e.g., 0.999). <=0 disables EMA")
    ap.add_argument("--use_ema_for_eval", type=int, default=None,
                    help="1 to use EMA weights for final save/inference, 0 to use raw weights")

    # ---------- Logging / outputs ----------
    ap.add_argument("--print_every", type=int, default=None)
    ap.add_argument("--model_out", type=str, default=None)
    ap.add_argument("--preds_out", type=str, default=None)

    # ---------- Checkpointing / resume / inference ----------
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume (weights only). If not set, will try ckpt_dir/last.pt")
    ap.add_argument("--ckpt_dir", type=str, default=None,
                    help="Directory to write checkpoints (default: checkpoints)")
    ap.add_argument("--save_every", type=int, default=None,
                    help="Save named checkpoint every N epochs (always writes rolling last.pt each epoch)")
    ap.add_argument("--run_inference", type=int, default=None,
                    help="1 to run inference at end of job (rank0), 0 to skip")

    # ---------- Debug ----------
    ap.add_argument("--detect_anomaly", action="store_true",
                    help="Enable autograd anomaly detection")

    return ap.parse_args()


def load_cfg(args):
    """
    Load JSON config (if provided), then override with CLI flags.
    Produce a dict matching the keys expected by ddp_pinn_helper.ddp_train_entry.
    """
    cfg = {}

    # 1) Load base config file if provided
    if args.config:
        if not os.path.isfile(args.config):
            raise SystemExit(f"ERROR: --config file not found: {args.config}")
        with open(args.config, "r") as f:
            try:
                cfg = json.load(f)
            except Exception as e:
                raise SystemExit(f"ERROR: failed to parse JSON config {args.config}: {e}")

    # Helper to override
    def set_if(key, val):
        if val is not None:
            cfg[key] = val

    # 2) Apply CLI overrides / mapping
    # dataset / epochs
    set_if("npz_path", args.npz)
    set_if("epochs", args.epochs)
    set_if("samples_per_epoch", args.samples_per_epoch)
    if args.reshuffle_across_ranks is not None:
        cfg["reshuffle_across_ranks"] = bool(args.reshuffle_across_ranks)

    # model / optimizer / collocation
    set_if("hidden", args.hidden)
    set_if("depth", args.depth)
    set_if("lr", args.lr)
    set_if("n_colloc", args.n_colloc)

    # supervised slice
    set_if("sup_points", args.sup_points)
    set_if("edge_supervision_frac", args.edge_supervision_frac)

    # physics / weights
    if args.accom is not None:
        cfg["accom_val"] = args.accom
    set_if("W_PHYS", args.w_phys)
    set_if("W_IC", args.w_ic)
    set_if("W_DATA", args.w_data)
    if args.ic_auto_balance is not None:
        cfg["ic_auto_balance"] = bool(args.ic_auto_balance)

    # scheduler / regularization / EMA
    if args.use_scheduler is not None:
        cfg["use_scheduler"] = int(args.use_scheduler)
    set_if("lr_factor", args.lr_factor)
    set_if("lr_patience", args.lr_patience)
    set_if("min_lr", args.min_lr)

    set_if("clip_grad_norm", args.clip_grad_norm)

    set_if("ema_decay", args.ema_decay)
    if args.use_ema_for_eval is not None:
        cfg["use_ema_for_eval"] = int(args.use_ema_for_eval)

    # logging / outputs
    set_if("print_every", args.print_every)
    set_if("model_out", args.model_out)
    set_if("preds_out", args.preds_out)

    # checkpoint / resume / inference
    if args.resume is not None:
        cfg["resume_from"] = args.resume
    set_if("ckpt_dir", args.ckpt_dir)
    set_if("save_every", args.save_every)
    if args.run_inference is not None:
        cfg["run_inference"] = int(args.run_inference)

    # debug
    if args.detect_anomaly:
        cfg["detect_anomaly"] = True

    # 3) Sanity & defaults
    if "npz_path" not in cfg or not cfg["npz_path"]:
        raise SystemExit("ERROR: dataset path missing. Provide it in --config under 'npz_path' "
                         "or pass --npz=/path/to/combined_outputs.npz")

    # Defaults (JSON/CLI takes precedence if set)
    cfg.setdefault("hidden", 256)
    cfg.setdefault("depth", 5)
    cfg.setdefault("epochs", 200)            # per job; your submitter can override with -E/--epochs
    cfg.setdefault("lr", 2e-4)
    cfg.setdefault("n_colloc", 64)

    cfg.setdefault("W_PHYS", 1.0)
    cfg.setdefault("W_IC",   1.0)
    cfg.setdefault("W_DATA", 0.1)
    cfg.setdefault("accom_val", 0.3)
    cfg.setdefault("kappa_val", 0.54)

    cfg.setdefault("sup_points", 96)
    cfg.setdefault("edge_supervision_frac", 0.30)
    cfg.setdefault("ic_auto_balance", True)

    cfg.setdefault("use_scheduler", 1)
    cfg.setdefault("lr_factor", 0.5)
    cfg.setdefault("lr_patience", 8)
    cfg.setdefault("min_lr", 1e-7)

    cfg.setdefault("clip_grad_norm", 0.5)

    cfg.setdefault("ema_decay", 0.999)
    cfg.setdefault("use_ema_for_eval", 1)

    cfg.setdefault("samples_per_epoch", 3200)       # divisible by 16 for 4 nodes x 4 GPUs
    cfg.setdefault("reshuffle_across_ranks", True)

    cfg.setdefault("print_every", 1)
    cfg.setdefault("model_out", "pinn.parcel.pt")
    cfg.setdefault("preds_out", "pinn_preds.npz")

    cfg.setdefault("ckpt_dir", "checkpoints")
    cfg.setdefault("save_every", 50)
    cfg.setdefault("run_inference", 1)

    # ddp_pinn_helper accepts 'resume_from' (if not set, it will try ckpt_dir/last.pt automatically)
    cfg.setdefault("resume_from", None)

    return cfg


def main():
    args = parse_args()
    cfg = load_cfg(args)

    # Light banner (rank-agnostic; actual per-rank logging happens in helper)
    print("[launcher] Using config:")
    printable = {k: v for k, v in cfg.items()}
    print(json.dumps(printable, indent=2, sort_keys=True))
    sys.stdout.flush()

    helper.ddp_train_entry(cfg)


if __name__ == "__main__":
    main()

