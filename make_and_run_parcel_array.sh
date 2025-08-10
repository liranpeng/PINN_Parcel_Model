#!/usr/bin/env bash
set -euo pipefail

# =========================
# User-tunable parameters
# =========================
PROJECT_ACCT="m4334"                 # SLURM account
QUEUE="regular"                         # SLURM queue/partition
TIME_LIMIT="48:00:00"                # wallclock
MAIL_USER="liranp@uci.edu"
ENV_ACTIVATE_CMD='conda activate mytorchenv'  # how to activate your env

# Sampling config
NSAMPLES=12288          # total samples (adjustable)
NTASKS=128               # number of MPI tasks / cores
SEED=42

# Output folders
WORKDIR="/pscratch/sd/h/heroplr/Parcel_model/Standalone/parcel_array_$(date +%Y%m%d_%H%M%S)"
OUTDIR_TASK="${WORKDIR}/task_outputs"
INPUTS_FILE="${WORKDIR}/inputs.npz"
COMBINED_OUT="${WORKDIR}/combined_outputs.npz"

# Ranges for random sampling (uniform)
# T0 [K], S0 [unitless], P0 [Pa], V [m/s]
T0_MIN=268.0; T0_MAX=290.0
S0_MIN=-0.02; S0_MAX=0.02
P0_MIN=70000.0; P0_MAX=90000.0
V_MIN=0.2; V_MAX=10.0

# Aerosol 1 and 2: (sigma, N) only — mu, kappa fixed; bins fixed
SIG1_MIN=1.4; SIG1_MAX=1.8
N1_MIN=100.0; N1_MAX=2000.0

SIG2_MIN=1.4; SIG2_MAX=1.8
N2_MIN=10.0;  N2_MAX=1500.0

# Fixed aerosol settings
MU1=0.0305
MU2=0.0894
KAPPA=0.54
BINS=200

# Parcel integration knobs
OUTPUT_DT=1.0
ACCOM=0.3

# ===========================================
# Make workspace
# ===========================================
mkdir -p "${WORKDIR}" "${OUTDIR_TASK}"
cd "${WORKDIR}"
cp /pscratch/sd/h/heroplr/Parcel_model/Standalone/parcel_model_standalone.py "${WORKDIR}"
cp /pscratch/sd/h/heroplr/Parcel_model/Standalone/aerosol_tools.py "${WORKDIR}"
echo "[INFO] Working directory: ${WORKDIR}"

# ===========================================
# 1) generate_inputs.py
# ===========================================
cat > generate_inputs.py << 'PY'
import numpy as np
import argparse
import os, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ranges", type=str, required=True)
    ap.add_argument("--mu1", type=float, required=True)
    ap.add_argument("--mu2", type=float, required=True)
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--bins", type=int, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    r = json.loads(args.ranges)

    def U(lo, hi, size):
        return rng.uniform(lo, hi, size=size).astype(float)

    n = int(args.n)
    T0 = U(r["T0_MIN"], r["T0_MAX"], n)
    S0 = U(r["S0_MIN"], r["S0_MAX"], n)
    P0 = U(r["P0_MIN"], r["P0_MAX"], n)
    V  = U(r["V_MIN"],  r["V_MAX"],  n)

    sig1 = U(r["SIG1_MIN"], r["SIG1_MAX"], n)
    N1   = U(r["N1_MIN"],   r["N1_MAX"],   n)
    sig2 = U(r["SIG2_MIN"], r["SIG2_MAX"], n)
    N2   = U(r["N2_MIN"],   r["N2_MAX"],   n)

    meta = dict(
        n=n,
        seed=args.seed,
        bins=args.bins,
        mu1=args.mu1, mu2=args.mu2,
        kappa=args.kappa,
        ranges=r
    )

    np.savez_compressed(
        args.out,
        T0=T0, S0=S0, P0=P0, V=V,
        sig1=sig1, N1=N1, sig2=sig2, N2=N2,
        meta=json.dumps(meta)
    )
    print(f"[OK] wrote {args.out} with {n} samples")

if __name__ == "__main__":
    main()
PY

# ===========================================
# 2) run_samples.py  (per-rank worker)
# ===========================================
cat > run_samples.py << 'PY'
import os, json, argparse
import numpy as np
from aerosol_tools import Lognorm, AerosolSpecies
from parcel_model_standalone import build_initial_state, run_parcel_model

def choose_chunk(n, ntasks, task_id):
    # Even partition with last chunk possibly shorter
    base = n // ntasks
    rem = n % ntasks
    start = task_id * base + min(task_id, rem)
    end = start + base + (1 if task_id < rem else 0)
    return start, end

def obj_array(lst):
    arr = np.empty(len(lst), dtype=object)
    for i, v in enumerate(lst): arr[i] = v
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--ntasks", type=int, required=True)
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--mu1", type=float, required=True)
    ap.add_argument("--mu2", type=float, required=True)
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--bins", type=int, required=True)
    ap.add_argument("--output-dt", type=float, default=1.0)
    ap.add_argument("--accom", type=float, default=0.3)
    args = ap.parse_args()

    # Detect SLURM rank if not passed
    if args.task_id is None:
        args.task_id = int(os.environ.get("SLURM_PROCID", "0"))

    data = np.load(args.inputs, allow_pickle=True)
    n = int(json.loads(str(data["meta"]))["n"])

    T0 = data["T0"]; S0 = data["S0"]; P0 = data["P0"]; Varr = data["V"]
    sig1 = data["sig1"]; N1 = data["N1"]; sig2 = data["sig2"]; N2 = data["N2"]

    start, end = choose_chunk(n, args.ntasks, args.task_id)
    idxs = np.arange(start, end, dtype=int)
    if len(idxs) == 0:
        # Nothing to do for this rank
        out_path = os.path.join(args.outdir, f"outputs_task_{args.task_id:03d}.npz")
        np.savez_compressed(out_path, idx=idxs)
        print(f"[TASK {args.task_id}] empty chunk -> wrote {out_path}")
        return

    y0_list, x_list, z_list, meta_list = [], [], [], []

    for i in idxs:
        # Build aerosol species for this sample
        a1 = AerosolSpecies(
            'sulfate',
            Lognorm(mu=args.mu1, sigma=float(sig1[i]), N=float(N1[i])),
            kappa=args.kappa,
            bins=args.bins
        )
        a2 = AerosolSpecies(
            'sulfate',
            Lognorm(mu=args.mu2, sigma=float(sig2[i]), N=float(N2[i])),
            kappa=args.kappa,
            bins=args.bins
        )
        aerosol = [a1, a2]

        # y0 and run
        y0, r_drys, Nis, kappas, nr, names = build_initial_state(
            aerosol, V=float(Varr[i]), T0=float(T0[i]), S0=float(S0[i]), P0=float(P0[i])
        )

        # Follow your example: t_end scales with V
        t_end = 500.0 / float(Varr[i])

        x, z, meta = run_parcel_model(
            y0=y0,
            aerosols=aerosol,
            V=float(Varr[i]),
            t_end=float(t_end),
            output_dt=float(args.output_dt),
            verbose=False,
            accom=float(args.accom)
        )

        y0_list.append(y0)
        x_list.append(x)
        z_list.append(z)
        meta_list.append(meta)

    # Save per-task object arrays (variable lengths -> use pickling)
    out_path = os.path.join(args.outdir, f"outputs_task_{args.task_id:03d}.npz")
    np.savez_compressed(
        out_path,
        idx=idxs,
        y0_list=obj_array(y0_list),
        x_list=obj_array(x_list),
        z_list=obj_array(z_list),
        meta_list=obj_array(meta_list)
    )
    print(f"[TASK {args.task_id}] wrote {out_path} with {len(idxs)} samples")

if __name__ == "__main__":
    main()
PY

# ===========================================
# 3) combine_outputs.py
# ===========================================
cat > combine_outputs.py << 'PY'
import os, argparse, glob
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "outputs_task_*.npz")))
    if not files:
        raise SystemExit("No per-task outputs found.")

    idx_all, y0_all, x_all, z_all, meta_all = [], [], [], [], []

    for f in files:
        d = np.load(f, allow_pickle=True)
        idx = d["idx"]
        if idx.size == 0:
            continue
        idx_all.append(idx)
        y0_all.extend(d["y0_list"] if "y0_list" in d else [])
        x_all.extend(d["x_list"] if "x_list" in d else [])
        z_all.extend(d["z_list"] if "z_list" in d else [])
        meta_all.extend(d["meta_list"] if "meta_list" in d else [])

    # Concatenate indices; other fields remain lists with variable shapes
    idx_all = np.concatenate(idx_all) if idx_all else np.array([], dtype=int)

    # Store as object arrays
    def to_obj(lst):
        arr = np.empty(len(lst), dtype=object)
        for i,v in enumerate(lst): arr[i]=v
        return arr

    np.savez_compressed(
        args.out,
        idx=idx_all,
        y0_list=to_obj(y0_all),
        x_list=to_obj(x_all),
        z_list=to_obj(z_all),
        meta_list=to_obj(meta_all)
    )
    print(f"[OK] combined -> {args.out} (samples={len(idx_all)})")

    if args.cleanup:
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
        print(f"[OK] cleaned up {len(files)} task files")

if __name__ == "__main__":
    main()
PY

# ===========================================
# 4) (Optional) Generate 128 tiny wrappers
#     — not required for SLURM run, but created as requested.
# ===========================================
mkdir -p workers
for i in $(seq -w 0 $((NTASKS-1))); do
cat > workers/worker_${i}.py <<PY
import sys
from run_samples import main
if __name__ == "__main__":
    sys.argv = [
        "run_samples.py",
        "--inputs","${INPUTS_FILE}",
        "--outdir","${OUTDIR_TASK}",
        "--ntasks","${NTASKS}",
        "--task-id","$((10#$i))",
        "--mu1","${MU1}",
        "--mu2","${MU2}",
        "--kappa","${KAPPA}",
        "--bins","${BINS}",
        "--output-dt","${OUTPUT_DT}",
        "--accom","${ACCOM}",
    ]
    main()
PY
done

# ===========================================
# 5) SLURM job file
# ===========================================
cat > job_parcel_array.sbatch <<SB
#!/bin/bash
#SBATCH -N 1
#SBATCH -q ${QUEUE}
#SBATCH -t ${TIME_LIMIT}
#SBATCH -J parcel_array
#SBATCH -A ${PROJECT_ACCT}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL_USER}
#SBATCH -C cpu
#SBATCH --ntasks=${NTASKS}
#SBATCH -L scratch,cfs

module load python || true
source ~/.bashrc || true
conda activate mytorchenv
conda env list
cd "${WORKDIR}"
pwd

echo "[INFO] Generating inputs..."
python generate_inputs.py \\
  --out "${INPUTS_FILE}" \\
  --n ${NSAMPLES} \\
  --seed ${SEED} \\
  --ranges '{
    "T0_MIN": ${T0_MIN}, "T0_MAX": ${T0_MAX},
    "S0_MIN": ${S0_MIN}, "S0_MAX": ${S0_MAX},
    "P0_MIN": ${P0_MIN}, "P0_MAX": ${P0_MAX},
    "V_MIN": ${V_MIN},   "V_MAX": ${V_MAX},
    "SIG1_MIN": ${SIG1_MIN}, "SIG1_MAX": ${SIG1_MAX},
    "N1_MIN": ${N1_MIN},     "N1_MAX": ${N1_MAX},
    "SIG2_MIN": ${SIG2_MIN}, "SIG2_MAX": ${SIG2_MAX},
    "N2_MIN": ${N2_MIN},     "N2_MAX": ${N2_MAX}
  }' \\
  --mu1 ${MU1} --mu2 ${MU2} --kappa ${KAPPA} --bins ${BINS}

echo "[INFO] Launching ${NTASKS} tasks with srun..."
srun -n ${NTASKS} python run_samples.py \\
  --inputs "${INPUTS_FILE}" \\
  --outdir "${OUTDIR_TASK}" \\
  --ntasks ${NTASKS} \\
  --mu1 ${MU1} --mu2 ${MU2} --kappa ${KAPPA} --bins ${BINS} \\
  --output-dt ${OUTPUT_DT} --accom ${ACCOM}

echo "[INFO] Combining per-task outputs..."
python combine_outputs.py --indir "${OUTDIR_TASK}" --out "${COMBINED_OUT}" --cleanup

echo "[DONE] Combined file at: ${COMBINED_OUT}"
SB

# ===========================================
# Submit the job
# ===========================================
echo "[INFO] Submitting SLURM job..."
sbatch job_parcel_array.sbatch

echo "[INFO] Setup complete."
echo "Workdir: ${WORKDIR}"
echo "You can monitor with: squeue -u \$USER"


