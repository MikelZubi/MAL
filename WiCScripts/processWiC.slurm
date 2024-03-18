#!/bin/bash
#SBATCH --job-name=WiC_LLMs
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:2
#SBATCH --output=./SLURM/LOG/irteeraWiC_LLMs1.log
#SBATCH --error=./SLURM/ERRORE/erroreWiC_LLMs1.err

source ~/inguruneak/GRAL/bin/activate

#echo "Gemma7"
#srun python WiCScripts/processWiC.py Gemma7 0
#echo "Gemma2"
#srun python WiCScripts/processWiC.py Gemma2 0
#echo "Zephyr"
#srun python WiCScripts/processWiC.py Zephyr 0
#echo "Mistral"
#srun python WiCScripts/processWiC.py Mistral 0
#echo "Llama2"
#srun python WiCScripts/processWiC.py Llama2 0
echo "Mixtral"
#srun python WiCScripts/processWiC.py Mixtral 0
srun python WiCScripts/processWiC.py Mixtral 1
srun python WiCScripts/processWiC.py Mixtral 2
srun python WiCScripts/processWiC.py Mixtral 3
srun python WiCScripts/processWiC.py Mixtral 5
srun python WiCScripts/processWiC.py Mixtral 7

echo "MixtralDPO"
srun python WiCScripts/processWiC.py MixtralDPO 0
srun python WiCScripts/processWiC.py MixtralDPO 1
srun python WiCScripts/processWiC.py MixtralDPO 2
srun python WiCScripts/processWiC.py MixtralDPO 3
srun python WiCScripts/processWiC.py MixtralDPO 5
srun python WiCScripts/processWiC.py MixtralDPO 7







