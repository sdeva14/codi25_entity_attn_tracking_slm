#!/bin/bash

#SBATCH -p ice-deep.p
#SBATCH --job-name=attn_ent
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

log_dir=$1
mkdir -p $log_dir

source ~/miniconda3/bin/activate chat

logpath="./exp/${log_dir}"
mkdir -p $logpath
logfile="$logpath/job_${SLURM_JOB_ID}.out"

sample_num=$2
encoder_weights=$3

python attn_flow_test.py exp_args.sample_num=${sample_num} model.llm_id=${encoder_weights} > ${logfile}
