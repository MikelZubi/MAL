#!/bin/bash
#SBATCH --job-name=WiC_LLMs+Exp
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:2
#SBATCH --output=./SLURM/LOG/irteeraWiC_LLMs+Exp.log
#SBATCH --error=./SLURM/ERRORE/erroreWiC_LLMs+Exp.err

source ~/inguruneak/GRAL/bin/activate


echo "Few"
echo "Mixtral"
srun python WiCScripts/WiCDefExamples.py modeloak/Mixtral-8x7B-Instruct-v0.1 WiCOutputs/FewShot+Exp/Mixtral.json Few
#echo "ZephyrB"
#srun python WiCScripts/WiCDefExamples.py modeloak/zephyr-7b-beta WiCOutputs/FewShot+Exp/ZephyrB.json Few

echo "Zero"
echo "Mixtral"
srun python WiCScripts/WiCDefExamples.py modeloak/Mixtral-8x7B-Instruct-v0.1 WiCOutputs/ZeroShot+Exp/Mixtral.json Zero
#echo "ZephyrB"
#srun python WiCScripts/WiCDefExamples.py modeloak/zephyr-7b-beta WiCOutputs/ZeroShot+Exp/ZephyrB.json Zero


