#!/usr/bin/env bash
set -euo pipefail

# sulab7g-zxy direct launcher for Stage 2 projection-head fallback.
# Run only after the no-projection sweep if metrics/motif support do not improve.

REPO_ROOT=${REPO_ROOT:-/PROJ5/liangn_zxy/work/expression}
PYTHON=${PYTHON:-/PROJ5/liangn_zxy/envs/promodel/bin/python}
RUN_ROOT=${STAGE2_RUN_ROOT:-/PROJ5/liangn_zxy/runs/expression/stage2}
DATA=${STAGE2_DATA:-promoter_stage2_v1}
SCRNA=${STAGE2_SCRNA:-data/umi_E-MTAB-10519-hqcells/integrated_data.h5ad}
CELL_SPLIT_DIR=${STAGE2_CELL_SPLIT_DIR:-data/${DATA}}
INPUT_PANEL=${STAGE2_INPUT_PANEL:-data/${DATA}/input_gene_panel_train.txt}
GPU_CSV=${STAGE2_GPUS:-0,1,2}
CW=${STAGE2_PROJ_WEIGHT:-0.20}
PROJ_DIM=${STAGE2_PROJ_DIM:-64}
PROJ_PREFIX=${STAGE2_PROJ_PREFIX:-stage2_proj}

cd "${REPO_ROOT}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TMPDIR=${TMPDIR:-/PROJ5/liangn_zxy/scratch}
export HF_HOME=${HF_HOME:-/PROJ5/liangn_zxy/cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/PROJ5/liangn_zxy/cache/torch}
mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/launcher_logs" "${RUN_ROOT}/pids" "${TMPDIR}" "${HF_HOME}" "${TORCH_HOME}" outputs

if [ -e outputs/stage2 ] && [ ! -L outputs/stage2 ]; then
  echo "ERROR: outputs/stage2 exists and is not a symlink. Move it before launching." >&2
  exit 1
fi
ln -sfn "${RUN_ROOT}" outputs/stage2

IFS=',' read -r -a GPUS <<< "${GPU_CSV}"
SEEDS=(1 7 42)

launch_train() {
  local gpu=$1
  local seed=$2
  local cw_tag
  cw_tag=$("${PYTHON}" - <<PY
print(f"{int(round(float('${CW}') * 100)):03d}")
PY
)
  local exp_id="${PROJ_PREFIX}_cw${cw_tag}_proj${PROJ_DIM}_seed${seed}"
  local log_file="${RUN_ROOT}/launcher_logs/${exp_id}.log"
  local mutation_log_file="${RUN_ROOT}/launcher_logs/${exp_id}.mutation.log"
  echo "[$(date '+%F %T')] launch ${exp_id} gpu=${gpu} cw=${CW} proj=${PROJ_DIM} seed=${seed} log=${log_file}"
  (
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train.py \
      --exp_name "stage2/${exp_id}" \
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
      --loss combined \
      --pearson-lambda 5.0 \
      --nonzero-loss-weight 2.0 \
      --fusion gate \
      --expression-layer logcpm \
      --expression-transform none \
      --target-count-layer counts \
      --target-value-layer logcpm \
      --target-transform none \
      --checkpoint-metric val_rmse \
      --run-test-after-train \
      --test-max-samples 0 \
      --test-spearman-samples 0 \
      --samples-per-epoch 128000 \
      --val-samples 128000 \
      --cell-ratio 1.0 \
      --val-cell-ratio 1.0 \
      --max-duplication 1.0 \
      --gpu-cache-dataset \
      --gpu-sampler balanced \
      --batch-size 512 \
      --hidden-size 128 \
      --learning-rate 5e-4 \
      --warmup-epochs 5 \
      --eval-every-steps 512 \
      --patience 32 \
      --min-delta 1e-4 \
      --ema-alpha 0.9 \
      --epochs 80 \
      --num-workers 2 \
      --prefetch-factor 2 \
      --seed "${seed}" \
      > "${log_file}" 2>&1
    train_status=$?
    if [ "${train_status}" -ne 0 ]; then
      echo "[$(date '+%F %T')] training failed for ${exp_id}; mutation test skipped (status=${train_status})" >> "${mutation_log_file}"
      exit "${train_status}"
    fi
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
      --motif-window-size 9 \
      --motif-top-windows 200 \
      --motif-top-k 20 \
      --motif-min-support 3 \
      --batch-size 512 \
      --mutation-batch-size 512 \
      --num-workers 2 \
      > "${mutation_log_file}" 2>&1
  ) &
  echo $! > "${RUN_ROOT}/pids/${exp_id}.pid"
}

batch_pids=()
for i in "${!SEEDS[@]}"; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  launch_train "${gpu}" "${SEEDS[$i]}"
  batch_pids+=("$!")
  if [ "$(( (i + 1) % ${#GPUS[@]} ))" -eq 0 ]; then
    wait "${batch_pids[@]}"
    batch_pids=()
  fi
done
if [ "${#batch_pids[@]}" -gt 0 ]; then
  wait "${batch_pids[@]}"
fi

echo "[$(date '+%F %T')] Stage 2 projection fallback complete."
