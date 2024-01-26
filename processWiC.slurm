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


echo "Few"
echo "Mixtral"
srun python processWiC.py modeloak/Mixtral-8x7B-Instruct-v0.1 WiCOutputs/FewShot/Mixtral.json Few
echo "ZephyrB"
srun python processWiC.py modeloak/zephyr-7b-beta WiCOutputs/FewShot/ZephyrB.json Few

echo "Zero"
echo "Mixtral"
srun python processWiC.py modeloak/Mixtral-8x7B-Instruct-v0.1 WiCOutputs/ZeroShot/Mixtral.json Zero
echo "ZephyrB"
srun python processWiC.py modeloak/zephyr-7b-beta WiCOutputs/ZeroShot/ZephyrB.json Zero


