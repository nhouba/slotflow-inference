#!/bin/bash
#SBATCH --job-name=slotflow_chain
#SBATCH --account=a157
#SBATCH --partition=normal
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=360G
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=append

FLOW_DEPTH=8
MAX_K=10
N_TRAIN=10000000
SEED=42
PROJECT_DIR=/users/nhouba/SlotFlow
OUT_DIR=${PROJECT_DIR}/results
CKPT_DIR=${OUT_DIR}/depth_${FLOW_DEPTH}/checkpoints
mkdir -p "${OUT_DIR}" "${CKPT_DIR}" logs

# ================================================================
# 1. Handle timeout signal (outer shell)
# ================================================================
handle_timeout() {
    echo "[INFO] Time limit approaching — requeuing job ${SLURM_JOB_ID} ..."
    scontrol requeue ${SLURM_JOB_ID}
    echo "[INFO] Requeue command sent."
    exit 0
}
trap handle_timeout USR1

# ================================================================
# 2. Launch environment and training (env vars set *inside*)
# ================================================================
uenv run --view default pytorch/v2.6.0:v1 bash <<'EOF'
set -e  # exit immediately on error
source ~/clariden_gpu_env/bin/activate
cd /users/nhouba/SlotFlow || exit 1

# Re-declare environment variables inside heredoc (since outer vars aren’t inherited)
FLOW_DEPTH=8
MAX_K=10
N_TRAIN=10000000
SEED=42
OUT_DIR=/users/nhouba/SlotFlow/results

echo "[INFO] ================================================================"
echo "[INFO] Job started on $(hostname) at $(date)"
echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Working directory: $(pwd)"
echo "[INFO] ================================================================"

CKPT_DIR="results/depth_${FLOW_DEPTH}/checkpoints"
mkdir -p "$CKPT_DIR"
RESUME_FILE=$(find "$CKPT_DIR" -maxdepth 1 -name "last.ckpt" | sort | tail -n 1)

# === NCCL environment tweaks for GH200 ===
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_NET="Socket"
export NCCL_SOCKET_IFNAME=^lo,docker,virbr,ib
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_LAUNCH_MODE=GROUP
export TORCH_NCCL_ENABLE_MONITORING=0

if [ -z "$RESUME_FILE" ]; then
    echo "[INFO] No checkpoint found — starting new training."
    srun python Train-cluster.py \
        --flow_depth ${FLOW_DEPTH} \
        --max_K ${MAX_K} \
        --N_train ${N_TRAIN} \
        --seed ${SEED} \
        --out_dir ${OUT_DIR}
else
    echo "[INFO] Found checkpoint: $RESUME_FILE"
    srun python Train-cluster.py \
        --flow_depth ${FLOW_DEPTH} \
        --max_K ${MAX_K} \
        --N_train ${N_TRAIN} \
        --seed ${SEED} \
        --out_dir ${OUT_DIR} \
        --resume_from "$RESUME_FILE"
fi

echo "[INFO] Job finished at $(date)"
EOF
