#!/bin/bash
#SBATCH --job-name=WiC_LLMs
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:2
#SBATCH --output=./SLURM/LOG/irteeraArtExamples_LLMs1.log
#SBATCH --error=./SLURM/ERRORE/erroreArtExamples_LLMs1.err

source ~/inguruneak/GRAL/bin/activate


srun python WiCScripts/examplesWiC.py Zephyr 4 1
srun python WiCScripts/examplesWiC.py Zephyr 4 2
srun python WiCScripts/examplesWiC.py Zephyr 4 3