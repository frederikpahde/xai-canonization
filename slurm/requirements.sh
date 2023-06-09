#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=requirements
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

source "/etc/slurm/local_job_dir.sh"

mkdir -p ${LOCAL_JOB_DIR}/output

singularity \
  run \
        --nv \
        --bind ${LOCAL_JOB_DIR}/output:/mnt/output \
        ../singularity/requirements.sif

cd ${LOCAL_JOB_DIR}
tar -czf requirements.tgz output
cp requirements.tgz ${SLURM_SUBMIT_DIR}

rm -rf ${LOCAL_JOB_DIR}/*
