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

huggingface-cli login --token

echo "Llama3"
srun python WiCScripts/LLMsWiC.py Llama3 0 Oxford
srun python WiCScripts/LLMsWiC.py Llama3 1 Oxford
srun python WiCScripts/LLMsWiC.py Llama3 2 Oxford
srun python WiCScripts/LLMsWiC.py Llama3 3 Oxford
srun python WiCScripts/LLMsWiC.py Llama3 4 Oxford
srun python WiCScripts/LLMsWiC.py Llama3 5 Oxford

python WiCScripts/makeTable.py Oxford
 


