#!/bin/bash

#SBATCH -p ice-deep.p

#SBATCH --job-name=attn_ent    # Job name
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
##SBATCH -o ./exp/job_%j.out
##SBATCH -e ./exp/err_%j.out

## nodes allocation
#SBATCH --nodes=1               # number of nodes
##SBATCH --ntasks-per-node=1     # MPI processes per node
#SBATCH --ntasks=1                    # should be the same as the number of gpus
#SBATCH --gres=gpu:1            # number of GPUs per node (gres=gpu:N)
## #SBATCH --gpu-bind=single:1     # bind each process to its own GPU (single:<tasks_per_gpu>)
## #SBATCH --gpus-per-task=1       # number of GPUs per process

log_dir=$1
mkdir -p $log_dir

# echo "Writing to ${logfile}"
# scontrol show -dd job $SLURM_JOB_ID
# printenv

# ml use /hits/sw/shared/ice-deep/a89/modules/all
# module load CUDA/11.8.0
source ~/miniconda3/bin/activate chat

logpath="./exp/${log_dir}"
mkdir -p $logpath
logfile="$logpath/job_${SLURM_JOB_ID}.out"

sample_num=$2
encoder_weights=$3

# cd ..
python attn_flow_test.py exp_args.sample_num=${sample_num} model.llm_id=${encoder_weights}  > ${logfile}

# python -m torch.distributed.launch --nproc_per_node=$num_gpus \
# examples/text-classification/run_glue.py \
# --model_name_or_path microsoft/deberta-v2-xxlarge \