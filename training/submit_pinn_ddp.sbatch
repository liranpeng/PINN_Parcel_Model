#!/bin/bash
#SBATCH -J pinn_ddp_2day
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 26                      # number of nodes
#SBATCH --ntasks-per-node=4          # <-- one Slurm task per GPU
#SBATCH --gpus-per-task=1            # <-- each task gets one GPU
#SBATCH --cpus-per-task=8            # 8 CPU threads per rank is plenty
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# -------- User knobs (override at submit time: CONFIG=... sbatch ...) --------
: "${CONFIG:=config.json}"        # Preferred JSON config
: "${NPZ_PATH:=}"                 # Fallback if no CONFIG
: "${NPROC_PER_NODE:=4}"          # GPUs per node (Perlmutter GPU = 4)
: "${MASTER_PORT:=29500}"         # Rendezvous port
: "${EXE:=train_ddp.py}"          # Entry script
: "${EXTRA_FLAGS:=}"              # Extra args to your train script

mkdir -p logs

# -------- Sanity checks --------
module load pytorch/2.6.0 >/dev/null 2>&1 || true
if [[ ! -f "$EXE" ]]; then
  echo "ERROR: Cannot find $EXE in $(pwd)" >&2
  exit 1
fi

# Decide CONFIG vs NPZ
USE_CONFIG=0
if [[ -n "${CONFIG}" && -f "${CONFIG}" ]]; then
  USE_CONFIG=1
elif [[ -z "${NPZ_PATH}" ]]; then
  echo "ERROR: Neither CONFIG='${CONFIG}' (not found) nor NPZ_PATH provided." >&2
  exit 1
fi

# -------- Derive cluster params --------
NNODES=${SLURM_NNODES:-1}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)

# GPUs per node safety
if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
else
  DETECTED_GPUS=${NPROC_PER_NODE}
fi
if (( NPROC_PER_NODE > DETECTED_GPUS )); then
  echo "WARN: NPROC_PER_NODE=${NPROC_PER_NODE} > detected ${DETECTED_GPUS}; clipping."
  NPROC_PER_NODE=${DETECTED_GPUS}
fi

# NCCL / Libfabric hints for Perlmutter
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn}
export FI_PROVIDER=cxi
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_CQ_CACHE=1
export CUDA_LAUNCH_BLOCKING=0

# PyTorch rendezvous
export MASTER_ADDR MASTER_PORT

# Build training args (quoted with = to avoid parsing issues)
TRAIN_ARGS=()
if (( USE_CONFIG == 1 )); then
  TRAIN_ARGS+=( "--config=${CONFIG}" )
else
  TRAIN_ARGS+=( "--npz=${NPZ_PATH}" )
fi
if [[ -n "${EXTRA_FLAGS}" ]]; then
  # shellcheck disable=SC2206
  TRAIN_ARGS+=( ${EXTRA_FLAGS} )
fi

echo "=============================="
echo " Batch DDP launch"
echo "  Nodes           : ${NNODES}"
echo "  GPUs / node     : ${NPROC_PER_NODE}"
echo "  Master          : ${MASTER_ADDR}:${MASTER_PORT}"
if (( USE_CONFIG == 1 )); then
  echo "  Config file     : ${CONFIG} (used)"
else
  echo "  NPZ path        : ${NPZ_PATH}"
fi
echo "  Entry           : ${EXE}"
echo "  Train args      : ${TRAIN_ARGS[*]}"
echo "=============================="
echo "Torch: $(python -c 'import torch; print(torch.__version__)')"

# -------- Launch: one srun task per node; each runs torchrun with 4 ranks --------
srun \
  --ntasks-per-node=1 \
  --gpus-per-task="${NPROC_PER_NODE}" \
  --cpu-bind=cores \
  bash -lc "
    module load pytorch/2.6.0 >/dev/null 2>&1 || true
    torchrun \
      --nproc_per_node=${NPROC_PER_NODE} \
      --nnodes=${NNODES} \
      --rdzv_backend=c10d \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      ${EXE} ${TRAIN_ARGS[@]}
  "

