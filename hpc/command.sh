#!/bin/bash
#SBATCH -J spatial
#SBATCH -p gpu_4l
#SBATCH -N 1
#SBATCH -A jchamper_g1
#SBATCH --qos jchamperg4c
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --gres=gpu:1
#SBATCH --overcommit
#SBATCH --mincpus=1
#SBATCH --no-requeue

source /home/jchamper_pkuhpc/miniconda3/etc/profile.d/conda.sh
conda activate promodel

echo "===== Job Info ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE"
echo "===== Python & PyTorch Info ====="
echo "Python: $(which python)"
python --version
python -m scripts.train > outputs.out




