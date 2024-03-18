#!/bin/bash
#SBATCH --job-name=WiC_LLMs
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM/LOG/irteeraArtExamples_LLMs1.log
#SBATCH --error=./SLURM/ERRORE/erroreArtExamples_LLMs1.err

source ~/inguruneak/GRAL/bin/activate


srun python WiCScripts/examplesWiC.py Llama2 8 0
srun python WiCScripts/examplesWiC.py Llama2 8 1
srun python WiCScripts/examplesWiC.py Llama2 8 2

srun python WiCScripts/examplesWiC.py Mistral 8 0
srun python WiCScripts/examplesWiC.py Mistral 8 1
srun python WiCScripts/examplesWiC.py Mistral 8 2