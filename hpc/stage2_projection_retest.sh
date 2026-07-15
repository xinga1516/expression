#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate completed projection-head checkpoints after fixing model_test loading.
REPO_ROOT=${REPO_ROOT:-/PROJ5/liangn_zxy/work/expression}
PYTHON=${PYTHON:-/PROJ5/liangn_zxy/envs/promodel/bin/python}
RUN_ROOT=${STAGE2_RUN_ROOT:-/PROJ5/liangn_zxy/runs/expression/stage2}
DATA=${STAGE2_DATA:-promoter_stage2_v1}
SCRNA=${STAGE2_SCRNA:-data/umi_E-MTAB-10519-hqcells/integrated_data.h5ad}
CELL_SPLIT_DIR=${STAGE2_CELL_SPLIT_DIR:-data/${DATA}}
INPUT_PANEL=${STAGE2_INPUT_PANEL:-data/${DATA}/input_gene_panel_train.txt}
GPU_CSV=${STAGE2_GPUS:-0,1,2}
PREFIX=${STAGE2_PROJ_PREFIX:-stage2_proj_cw040_proj64}
SEEDS=(1 7 42)

cd "${REPO_ROOT}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TMPDIR=${TMPDIR:-/PROJ5/liangn_zxy/scratch}
mkdir -p "${RUN_ROOT}/launcher_logs" "${TMPDIR}"

IFS=',' read -r -a GPUS <<< "${GPU_CSV}"

run_one() {
  local gpu=$1
  local seed=$2
  local exp_id="${PREFIX}_seed${seed}"
  local log_file="${RUN_ROOT}/launcher_logs/${exp_id}.retest.log"
  echo "[$(date '+%F %T')] retest ${exp_id} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/model_test.py \
    --exp_name "stage2/${exp_id}" \
    --data "${DATA}" \
    --scrna-file "${SCRNA}" \
    --sequence-column sequence \
    --sequence-length 400 \
    --input-gene-panel-file "${INPUT_PANEL}" \
    --use-cell-split \
    --cell-split-dir "${CELL_SPLIT_DIR}" \
    --batch-size 512 \
    --mutation-batch-size 512 \
    --max-samples 0 \
    --spearman-samples 0 \
    --run-mutagenesis \
    --top-n 1000 \
    --max-pairs-per-gene-ratio 0.02 \
    --max-pairs-per-gene 20 \
    --motif-window-size 9 \
    --motif-top-windows 200 \
    --motif-top-k 20 \
    --motif-min-support 3 \
    --num-workers 2 \
    > "${log_file}" 2>&1
}

jobs=()
for i in "${!SEEDS[@]}"; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  run_one "${gpu}" "${SEEDS[$i]}" &
  jobs+=("$!")
done

for job in "${jobs[@]}"; do
  wait "${job}"
done

echo "[$(date '+%F %T')] projection checkpoint retest complete."
