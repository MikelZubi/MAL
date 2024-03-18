#!/bin/bash
#SBATCH --job-name=WiC_LLMs+Exp
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM/LOG/irteeraWiC_LLMs+Exp.log
#SBATCH --error=./SLURM/ERRORE/erroreWiC_LLMs+Exp.err

source ~/inguruneak/GRAL/bin/activate

mkdir -p WiCOutputs/0Shot+Exp
srun python WiCScripts/WiCDefExamples.py modeloak/Mistral-7B-Instruct-v0.2 WiCOutputs/0Shot+Exp/Mistral.json Mistral 0

# mkdir -p WiCOutputs/1Shot+Exp
# srun python WiCScripts/WiCDefExamples.py modeloak/Mistral-7B-Instruct-v0.2 WiCOutputs/1Shot+Exp/Mistral.json Mistral 1

# mkdir -p WiCOutputs/2Shot+Exp
# srun python WiCScripts/WiCDefExamples.py modeloak/Mistral-7B-Instruct-v0.2 WiCOutputs/2Shot+Exp/Mistral.json Mistral 2

# mkdir -p WiCOutputs/3Shot+Exp
# srun python WiCScripts/WiCDefExamples.py modeloak/Mistral-7B-Instruct-v0.2 WiCOutputs/3Shot+Exp/Mistral.json Mistral 3

# mkdir -p WiCOutputs/5Shot+Exp
# srun python WiCScripts/WiCDefExamples.py modeloak/Mistral-7B-Instruct-v0.2 WiCOutputs/5Shot+Exp/Mistral.json Mistral 5

# mkdir -p WiCOutputs/10Shot+Exp
# srun python WiCScripts/WiCDefExamples.py modeloak/Mistral-7B-Instruct-v0.2 WiCOutputs/10Shot+Exp/Mistral.json Mistral 10

# mkdir -p WiCOutputs/15Shot+Exp
# srun python WiCScripts/WiCDefExamples.py modeloak/Mistral-7B-Instruct-v0.2 WiCOutputs/15Shot+Exp/Mistral.json Mistral 15

# mkdir -p WiCOutputs/20Shot+Exp
# srun python WiCScripts/WiCDefExamples.py modeloak/Mistral-7B-Instruct-v0.2 WiCOutputs/20Shot+Exp/Mistral.json Mistral 20

# mkdir -p WiCOutputs/25Shot+Exp
# srun python WiCScripts/WiCDefExamples.py modeloak/Mistral-7B-Instruct-v0.2 WiCOutputs/25Shot+Exp/Mistral.json Mistral 25

# mkdir -p WiCOutputs/30Shot+Exp
# srun python WiCScripts/WiCDefExamples.py modeloak/Mistral-7B-Instruct-v0.2 WiCOutputs/30Shot+Exp/Mistral.json Mistral 30


#srun python WiCScripts/WiCDefExamples.py modeloak/Mixtral-8x7B-Instruct-v0.1 WiCOutputs/FewShot+Exp/Mixtral.json Few
#echo "Mistral"
#srun python WiCScripts/WiCDefExamples.py modeloak/Mistral-7B-Instruct-v0.2 WiCOutputs/FewShot+Exp/Mistral.json Few


#srun python WiCScripts/WiCDefExamples.py modeloak/Mixtral-8x7B-Instruct-v0.1 WiCOutputs/ZeroShot+Exp/Mixtral.json Zero
#echo "Mistral"
#srun python WiCScripts/WiCDefExamples.py modeloak/Mistral-7B-Instruct-v0.2 WiCOutputs/ZeroShot+Exp/Mistral.json Zero

python WiCScripts/errorsTable.py Mistral WiCOutputs/MistralErrorTable.csv


