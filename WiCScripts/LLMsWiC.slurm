#!/bin/bash
#SBATCH --job-name=WiC_LLMs
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM/LOG/irteeraWiC_LLMsA1.log
#SBATCH --error=./SLURM/ERRORE/erroreWiC_LLMsA1.err

source ~/inguruneak/GRAL/bin/activate


echo "Zephyr"
srun python WiCScripts/LLMsWiC.py Zephyr 0
srun python WiCScripts/LLMsWiC.py Zephyr 1
srun python WiCScripts/LLMsWiC.py Zephyr 2
srun python WiCScripts/LLMsWiC.py Zephyr 3
srun python WiCScripts/LLMsWiC.py Zephyr 4
srun python WiCScripts/LLMsWiC.py Zephyr 5
srun python WiCScripts/LLMsWiC.py Zephyr 6
srun python WiCScripts/LLMsWiC.py Zephyr 7

echo "MistralDPO"
srun python WiCScripts/LLMsWiC.py Mistral 0
srun python WiCScripts/LLMsWiC.py Mistral 1
srun python WiCScripts/LLMsWiC.py Mistral 2
srun python WiCScripts/LLMsWiC.py Mistral 3
srun python WiCScripts/LLMsWiC.py Mistral 4
srun python WiCScripts/LLMsWiC.py Mistral 5
srun python WiCScripts/LLMsWiC.py Mistral 6
srun python WiCScripts/LLMsWiC.py Mistral 7

echo "MistralDPO"
srun python WiCScripts/LLMsWiC.py MistralDPO 0
srun python WiCScripts/LLMsWiC.py MistralDPO 1
srun python WiCScripts/LLMsWiC.py MistralDPO 2
srun python WiCScripts/LLMsWiC.py MistralDPO 3
srun python WiCScripts/LLMsWiC.py MistralDPO 4
srun python WiCScripts/LLMsWiC.py MistralDPO 5
srun python WiCScripts/LLMsWiC.py MistralDPO 6
srun python WiCScripts/LLMsWiC.py MistralDPO 7


echo "Llama2"
srun python WiCScripts/LLMsWiC.py Llama2 0
srun python WiCScripts/LLMsWiC.py Llama2 1
srun python WiCScripts/LLMsWiC.py Llama2 2
srun python WiCScripts/LLMsWiC.py Llama2 3
srun python WiCScripts/LLMsWiC.py Llama2 4
srun python WiCScripts/LLMsWiC.py Llama2 5
srun python WiCScripts/LLMsWiC.py Llama2 6
srun python WiCScripts/LLMsWiC.py Llama2 7

echo "OpenChat"
srun python WiCScripts/LLMsWiC.py OpenChat 0
srun python WiCScripts/LLMsWiC.py OpenChat 1
srun python WiCScripts/LLMsWiC.py OpenChat 2
srun python WiCScripts/LLMsWiC.py OpenChat 3
srun python WiCScripts/LLMsWiC.py OpenChat 4
srun python WiCScripts/LLMsWiC.py OpenChat 5
srun python WiCScripts/LLMsWiC.py OpenChat 6
srun python WiCScripts/LLMsWiC.py OpenChat 7