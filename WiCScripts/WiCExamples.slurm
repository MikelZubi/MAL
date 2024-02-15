#!/bin/bash
#SBATCH --job-name=WiC_SBs
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM/LOG/irteeraWiC_SB1.log
#SBATCH --error=./SLURM/ERRORE/erroreWiC_SB1.err

source ~/inguruneak/GRAL/bin/activate


echo "Sentence Bert"
srun python WiCScripts/WiCExamples.py WiCOutputs/OnlyExamples/SB.json


