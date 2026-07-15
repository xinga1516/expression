#!/usr/bin/env bash
set -euo pipefail

# Re-run Stage 2 mutation tests with the Stage 1-comparable parameter contract.
REPO_ROOT=${REPO_ROOT:-/PROJ5/liangn_zxy/work/expression}
PYTHON=${PYTHON:-/PROJ5/liangn_zxy/envs/promodel/bin/python}
RUN_ROOT=${STAGE2_RUN_ROOT:-/PROJ5/liangn_zxy/runs/expression/stage2}
DATA=${STAGE2_DATA:-promoter_stage2_v1}
SCRNA=${STAGE2_SCRNA:-data/umi_E-MTAB-10519-hqcells/integrated_data.h5ad}
CELL_SPLIT_DIR=${STAGE2_CELL_SPLIT_DIR:-data/${DATA}}
INPUT_PANEL=${STAGE2_INPUT_PANEL:-data/${DATA}/input_gene_panel_train.txt}
GPU_CSV=${STAGE2_GPUS:-0,1,2}
RUNS_TEXT=${STAGE2_MUTATION_RUNS:-"stage2_cw000_cw000_seed7 stage2_cw005_seed7 stage2_cw010_seed7 stage2_cw020_seed7 stage2_cw040_seed7 stage2_proj_cw040_proj64_seed7"}

cd "${REPO_ROOT}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TMPDIR=${TMPDIR:-/PROJ5/liangn_zxy/scratch}
mkdir -p "${RUN_ROOT}/launcher_logs" "${TMPDIR}"

IFS=',' read -r -a GPUS <<< "${GPU_CSV}"
read -r -a RUN_IDS <<< "${RUNS_TEXT}"

run_one() {
  local gpu=$1
  local exp_id=$2
  local log_file="${RUN_ROOT}/launcher_logs/${exp_id}.stage1_mutation_seed7.log"
  echo "[$(date '+%F %T')] Stage 1-comparable mutation: ${exp_id} gpu=${gpu}"
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
    --max-pairs-per-gene-ratio 0.02 \
    --max-pairs-per-gene 20 \
    --mutation-batch-size 512 \
    --motif-window-size 9 \
    --motif-top-windows 200 \
    --motif-top-k 20 \
    --motif-min-support 3 \
    --batch-size 512 \
    --num-workers 2 \
    > "${log_file}" 2>&1
}

jobs=()
for i in "${!RUN_IDS[@]}"; do
  run_one "${GPUS[$((i % ${#GPUS[@]}))]}" "${RUN_IDS[$i]}" &
  jobs+=("$!")
  if [ "${#jobs[@]}" -eq "${#GPUS[@]}" ]; then
    wait "${jobs[@]}"
    jobs=()
  fi
done
if [ "${#jobs[@]}" -gt 0 ]; then
  wait "${jobs[@]}"
fi

echo "[$(date '+%F %T')] Stage 1-comparable Stage 2 mutation retest complete."
