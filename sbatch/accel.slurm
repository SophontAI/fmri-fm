#!/bin/bash
#SBATCH --account=neurofoundation
#SBATCH --partition=pli
#SBATCH --job-name=found
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4       # Set to equal gres=gpu:#!
#SBATCH --mem=60G
#SBATCH --cpus-per-task=10      # Max is 12 CPUs per device for H100
#SBATCH --time=48:00:00         # total run time limit (HH:MM:SS) 
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --no-requeue
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ps6938@princeton.edu

nvidia-smi

export NUM_GPUS=4 # Set to equal gres=gpu:#!
echo NUM_GPUS=$NUM_GPUS

source ~/.bashrc
cd /scratch/gpfs/ps6938/fMRI-foundation-model/src

jupyter nbconvert main.ipynb --to python
if [ $? -ne 0 ]; then
  echo "Error: Conversion of ipynb to Python failed. Exiting."
  exit 1
fi

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000))
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=$((${SLURM_NNODES} * ${NUM_GPUS}))

NCCL_DEBUG=INFO

srun torchrun \
--nproc_per_node=$NUM_GPUS \
--nnodes=$SLURM_NNODES \
main.py
