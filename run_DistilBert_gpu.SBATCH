#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x.out
#SBATCH --job-name=torch
#SBATCH --mem=64GB

module purge


singularity exec --nv \
      	    --overlay /scratch/da3257/singularity/overlay-15GB-500K.ext3:rw \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python /scratch/da3257/hpml/project/lightning-ai/03_finetune-lora-script.py --q_lora False --lora_r 32 --lora_alpha 1 --lora_query True --lora_key False --lora_value True --lora_projection True --lora_mlp True --lora_head False"
