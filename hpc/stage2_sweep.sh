#!/bin/bash
#SBATCH -J promoter_stage2
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
#SBATCH -a 1-12

set -euo pipefail

# Run from repository root on sulab7g-zxy, for example:
#   mkdir -p outputs/stage2/slurm
#   sbatch hpc/stage2_sweep.sh
#
# This sweep assumes Stage 2 assets have wider sequence caches, e.g.:
#   python scripts/build_promoter_stage1_assets.py \
#     --source-data umi_E-MTAB-10519-hqcells \
#     --output-data promoter_stage2_v1 \
#     --gtf data/raw/dmel-all-r6.54.gtf \
#     --genome-fasta data/raw/dmel-all-chromosome-r6.54.fasta \
#     --promoter-window-length 440 \
#     --positive-shift-bp 20 \
#     --seed 42

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

WEIGHTS=(0.05 0.10 0.20 0.40)
SEEDS=(1 7 42)
TOTAL=$(( ${#WEIGHTS[@]} * ${#SEEDS[@]} ))
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt "$TOTAL" ]; then
    echo "Task ${TASK_ID} outside 1-${TOTAL}; exiting."
    exit 0
fi

IDX=$((TASK_ID - 1))
WEIGHT_IDX=$((IDX / ${#SEEDS[@]}))
SEED_IDX=$((IDX % ${#SEEDS[@]}))
CW=${WEIGHTS[$WEIGHT_IDX]}
SEED=${SEEDS[$SEED_IDX]}
CW_TAG=$(printf "%03d" "$(python - <<PY
print(int(round(float('${CW}') * 100)))
PY
)")
EXP_NAME="stage2/stage2_cw${CW_TAG}_seed${SEED}"

echo "===== Stage 2 contrastive sweep ====="
echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} task=${TASK_ID} cuda=${CUDA_VISIBLE_DEVICES:-none}"
echo "data=${DATA} exp=${EXP_NAME} contrastive_weight=${CW} seed=${SEED}"

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
