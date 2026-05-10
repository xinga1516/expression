#!/bin/bash
#SBATCH -J lstm_vs_cnn
#SBATCH -p gpu_4l
#SBATCH -N 1
#SBATCH -A jchamper_g1
#SBATCH --qos jchamperg4c
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --gres=gpu:1
#SBATCH --overcommit
#SBATCH --mincpus=8
#SBATCH --no-requeue
#SBATCH -a 1-4

source /home/jchamper_pkuhpc/miniconda3/etc/profile.d/conda.sh
conda activate promodel

echo "===== Job Info ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

export TMPDIR=/tmp
export PYTHONPATH="PYTHONPATH:$pwd"

# Common args
DATA="highquality"
EPOCHS=30
BATCH=128
LR=1e-4


if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "=============================="
    echo "1/4  LSTMmodel (LSTM + Attention)"
    echo "=============================="
    python scripts/train.py \
        --exp_name test_lstm_attn \
        --model LSTMmodel \
        --data $DATA \
        --epochs $EPOCHS \
        --batch-size $BATCH \
        --hidden-size 64 \
        --learning-rate $LR \
        --plot-loss
fi

if [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    echo ""
    echo "=============================="
    echo "2/4  ConvAttentionModel (CNN + Attention)"
    echo "=============================="
    python scripts/train.py \
        --exp_name test_cnn_attn \
        --model ConvAttentionModel \
        --data $DATA \
        --epochs $EPOCHS \
        --batch-size $BATCH \
        --hidden-size 128 \
        --learning-rate $LR \
        --plot-loss
fi

if [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    echo ""
    echo "=============================="
    echo "3/4  PromoterBaseline (promoter only)"
    echo "=============================="
    python scripts/train.py \
        --exp_name test_promoter_baseline \
        --model PromoterBaseline \
        --data $DATA \
        --epochs $EPOCHS \
        --batch-size $BATCH \
        --hidden-size 128 \
        --learning-rate $LR \
        --plot-loss
fi

if [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    echo ""
    echo "=============================="
    echo "4/4  ExpressionBaseline (expression only)"
    echo "=============================="
    python scripts/train.py \
        --exp_name test_expr_baseline \
        --model ExpressionBaseline \
        --data $DATA \
        --epochs $EPOCHS \
        --batch-size $BATCH \
        --hidden-size 128 \
        --learning-rate $LR \
        --plot-loss
fi

