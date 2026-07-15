#!/usr/bin/env bash
set -euo pipefail

# Backfill mutation position tests for Stage 2 checkpoints produced by the
# legacy launcher before the post-training mutation hook was added. The script
# can start while training is active: it waits for each run to finish and
# skips a run if another mutation process is already handling it.

REPO_ROOT=${REPO_ROOT:-/PROJ5/liangn_zxy/work/expression}
PYTHON=${PYTHON:-/PROJ5/liangn_zxy/envs/promodel/bin/python}
DATA=${STAGE2_DATA:-promoter_stage2_v1}
SCRNA=${STAGE2_SCRNA:-data/umi_E-MTAB-10519-hqcells/integrated_data.h5ad}
CELL_SPLIT_DIR=${STAGE2_CELL_SPLIT_DIR:-data/${DATA}}
INPUT_PANEL=${STAGE2_INPUT_PANEL:-data/${DATA}/input_gene_panel_train.txt}
RUN_ROOT=${STAGE2_RUN_ROOT:-/PROJ5/liangn_zxy/runs/expression/stage2}

cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}/launcher_logs" "${RUN_ROOT}/pids"

RUN_IDS=(
  stage2_cw005_seed1
  stage2_cw005_seed7
  stage2_cw005_seed42
  stage2_cw010_seed1
  stage2_cw010_seed7
  stage2_cw010_seed42
  stage2_cw020_seed1
  stage2_cw020_seed7
  stage2_cw020_seed42
  stage2_cw040_seed1
  stage2_cw040_seed7
  stage2_cw040_seed42
)
GPUS=(4 5 6 7)

wait_until_ready() {
  local exp_id=$1
  local checkpoint="${RUN_ROOT}/${exp_id}/checkpoints/best_model.safetensors"
  local mutation_output="${RUN_ROOT}/${exp_id}/test/sequence_mutagenesis/position_importance.csv"
  while true; do
    if [ -f "${mutation_output}" ]; then
      return 1
    fi
    if pgrep -f "scripts/model_test.py.*stage2/${exp_id}" >/dev/null 2>&1; then
      return 1
    fi
    if pgrep -f "scripts/train.py.*stage2/${exp_id}" >/dev/null 2>&1; then
      sleep 30
      continue
    fi
    if [ -f "${checkpoint}" ]; then
      return 0
    fi
    sleep 30
  done
}

run_mutation() {
  local gpu=$1
  local exp_id=$2
  local log_file="${RUN_ROOT}/launcher_logs/${exp_id}.mutation.log"
  echo "[$(date '+%F %T')] backfill mutation ${exp_id} gpu=${gpu} log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/model_test.py \
    --exp_name "stage2/${exp_id}" \
    --data "${DATA}" \
    --scrna-file "${SCRNA}" \
    --sequence-column sequence \
    --sequence-length 400 \
    --input-gene-panel-file "${INPUT_PANEL}" \
    --use-cell-split \
    --cell-split-dir "${CELL_SPLIT_DIR}" \
    --skip-standard-test \
    --run-mutagenesis \
    --top-n 1000 \
    --max-pairs-per-gene-ratio 0.1 \
    --max-pairs-per-gene 100 \
    --motif-window-size 9 \
    --motif-top-windows 200 \
    --motif-top-k 20 \
    --motif-min-support 3 \
    --batch-size 512 \
    --mutation-batch-size 512 \
    --num-workers 2 \
    > "${log_file}" 2>&1
}

worker() {
  local gpu=$1
  local slot=$2
  local index
  for ((index=slot; index<${#RUN_IDS[@]}; index+=${#GPUS[@]})); do
    exp_id=${RUN_IDS[$index]}
    if wait_until_ready "${exp_id}"; then
      run_mutation "${gpu}" "${exp_id}"
    fi
  done
}

pids=()
for slot in "${!GPUS[@]}"; do
  worker "${GPUS[$slot]}" "${slot}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "[$(date '+%F %T')] pending Stage 2 mutation backfill complete."
