#!/bin/bash
# Recommended single-GPU profile for a 24G GPU with weaker CPU/data bandwidth.
# Adjust partition/account directives for the target server before sbatch.

#SBATCH -J promodel_24g
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o %j.out
#SBATCH -e %j.err

set -euo pipefail

if [ -f /home/jchamper_pkuhpc/miniconda3/etc/profile.d/conda.sh ]; then
    source /home/jchamper_pkuhpc/miniconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "conda.sh not found; activate the promodel environment before running this script." >&2
    exit 1
fi
conda activate promodel

export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

EXP_NAME="${EXP_NAME:-server_lstm128_gate_b1024}"
DATA="${DATA:-umi_processed}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-500000}"
VAL_SAMPLES="${VAL_SAMPLES:-64000}"

echo "===== Job Info ====="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "EXP_NAME=$EXP_NAME DATA=$DATA BATCH_SIZE=$BATCH_SIZE"

python scripts/train.py \
    --exp_name "$EXP_NAME" \
    --data "$DATA" \
    --model LSTMmodel \
    --hidden-size 128 \
    --fusion gate \
    --loss zinb \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --preencode-promoters \
    --cell-ratio 0.1 \
    --val-cell-ratio 0.25 \
    --nonzero-ratio 0.5 \
    --samples-per-epoch "$SAMPLES_PER_EPOCH" \
    --val-samples "$VAL_SAMPLES" \
    --learning-rate 1e-4 \
    --epochs 30 \
    --amp
