#!/bin/bash
#SBATCH --job-name=WiC_LLMs
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM/LOG/irteeraWiC_LLMs1.log
#SBATCH --error=./SLURM/ERRORE/erroreWiC_LLMs1.err

source ~/inguruneak/GRAL/bin/activate

#echo "Few"
#echo "Mixtral"
#srun python WiCScripts/processWiC.py modeloak/Mixtral-8x7B-Instruct-v0.1 WiCOutputs/FewShot/Mixtral.json Few
#echo "ZephyrB"
#srun python WiCScripts/processWiC.py modeloak/zephyr-7b-beta WiCOutputs/FewShot/ZephyrB.json Few

#echo "Mixtral"
#srun python WiCScripts/processWiC.py modeloak/Mixtral-8x7B-Instruct-v0.1 WiCOutputs/ZeroShot/Mixtral.json Zero
echo "ZephyrB"
#srun python WiCScripts/processWiC.py modeloak/zephyr-7b-beta WiCOutputs/ZeroShot/ZephyrB.json Zephyr 0
#mkdir -p WiCOutputs/3Shot
#srun python WiCScripts/processWiC.py modeloak/zephyr-7b-beta WiCOutputs/3Shot/ZephyrB.json Zephyr 3
#mkdir -p WiCOutputs/5Shot
#srun python WiCScripts/processWiC.py modeloak/zephyr-7b-beta WiCOutputs/5Shot/ZephyrB.json Zephyr 5
#mkdir -p WiCOutputs/10Shot
#srun python WiCScripts/processWiC.py modeloak/zephyr-7b-beta WiCOutputs/10Shot/ZephyrB.json Zephyr 10
mkdir -p WiCOutputs/15Shot
srun python WiCScripts/processWiC.py modeloak/zephyr-7b-beta WiCOutputs/15Shot/ZephyrB.json Zephyr 15
mkdir -p WiCOutputs/20Shot
srun python WiCScripts/processWiC.py modeloak/zephyr-7b-beta WiCOutputs/20Shot/ZephyrB.json Zephyr 20
mkdir -p WiCOutputs/25Shot
srun python WiCScripts/processWiC.py modeloak/zephyr-7b-beta WiCOutputs/25Shot/ZephyrB.json Zephyr 25
mkdir -p WiCOutputs/30Shot
srun python WiCScripts/processWiC.py modeloak/zephyr-7b-beta WiCOutputs/30Shot/ZephyrB.json Zephyr 30

python WiCScripts/errorsTable.py ZephyrB WiCOutputs/ZephyrErrorTable.csv




