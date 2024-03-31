#!/bin/bash
#SBATCH --job-name=WiC_LLMs
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM/LOG/irteeraArtExamplesDef_LLMs1.log
#SBATCH --error=./SLURM/ERRORE/erroreArtExamplesDef_LLMs1.err

source ~/inguruneak/GRAL/bin/activate


srun python WiCScripts/examplesDefWiC.py Zephyr 4 1
srun python WiCScripts/examplesDefWiC.py Zephyr 4 2
srun python WiCScripts/examplesDefWiC.py Zephyr 4 3