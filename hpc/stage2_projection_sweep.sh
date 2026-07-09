#!/bin/bash
#SBATCH -J promoter_stage2_proj
#SBATCH -p gpu_4l
#SBATCH -N 1
#SBATCH -A jchamper_g1
#SBATCH --qos jchamperg4c
#SBATCH -o outputs/stage2/slurm/%A_%a.out
#SBATCH -e outputs/stage2/slurm/%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --overcommit
#SBATCH --mincpus=8
#SBATCH --no-requeue
#SBATCH -a 1-3

set -euo pipefail

# Projection-head fallback. Use after the no-projection weight sweep if metrics/motif support do not improve.
# Runs cw=0.20 with projection_dim=64 across seeds by default. Override with STAGE2_PROJ_WEIGHT/STAGE2_PROJ_DIM.

if [ -f /home/jchamper_pkuhpc/miniconda3/etc/profile.d/conda.sh ]; then
    source /home/jchamper_pkuhpc/miniconda3/etc/profile.d/conda.sh
    conda activate promodel
elif [ -f /home/xinyue/miniconda3/etc/profile.d/conda.sh ]; then
    source /home/xinyue/miniconda3/etc/profile.d/conda.sh
    conda activate promodel_wsl
fi

mkdir -p outputs/stage2/slurm
export TMPDIR=${TMPDIR:-/tmp}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

DATA=${STAGE2_DATA:-promoter_stage2_v1}
SCRNA=${STAGE2_SCRNA:-data/umi_E-MTAB-10519-hqcells/integrated_data.h5ad}
CELL_SPLIT_DIR=${STAGE2_CELL_SPLIT_DIR:-data/${DATA}}
INPUT_PANEL=${STAGE2_INPUT_PANEL:-data/${DATA}/input_gene_panel_train.txt}
CW=${STAGE2_PROJ_WEIGHT:-0.20}
PROJ_DIM=${STAGE2_PROJ_DIM:-64}
SEEDS=(1 7 42)
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
SEED=${SEEDS[$((TASK_ID - 1))]}
CW_TAG=$(printf "%03d" "$(python - <<PY
print(int(round(float('${CW}') * 100)))
PY
)")
EXP_NAME="stage2/stage2_cw${CW_TAG}_proj${PROJ_DIM}_seed${SEED}"

echo "===== Stage 2 projection fallback ====="
echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} task=${TASK_ID} data=${DATA} exp=${EXP_NAME} cw=${CW} proj=${PROJ_DIM} seed=${SEED}"

python scripts/train.py \
  --exp_name "${EXP_NAME}" \
  --data "${DATA}" \
  --scrna-file "${SCRNA}" \
  --model CNNFlattenPromoterModel \
  --sequence-column sequence \
  --sequence-length 400 \
  --promoter-shift-max 20 \
  --contrastive-negative-shift-max -1 \
  --contrastive-weight "${CW}" \
  --contrastive-margin 1.0 \
  --contrastive-projection-dim "${PROJ_DIM}" \
  --contrastive-projection-layers 2 \
  --contrastive-positive-column positive_sequence \
  --contrastive-negative-column control_sequence \
  --use-cell-split \
  --cell-split-dir "${CELL_SPLIT_DIR}" \
  --input-gene-panel-file "${INPUT_PANEL}" \
  --loss mse \
  --checkpoint-metric val_rmse \
  --run-test-after-train \
  --test-spearman-samples 0 \
  --batch-size 512 \
  --hidden-size 128 \
  --learning-rate 5e-4 \
  --warmup-epochs 5 \
  --patience 32 \
  --ema-alpha 0.9999 \
  --epochs 30 \
  --num-workers 2 \
  --prefetch-factor 2 \
  --amp \
  --seed "${SEED}"
