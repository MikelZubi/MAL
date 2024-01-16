#!/bin/bash
#SBATCH --job-name=Test_LLMs
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --gres=gpu:2
#SBATCH --output=./SLURM/LOG/irteeraTest_LLMs1.log
#SBATCH --error=./SLURM/ERRORE/erroreTest_LLMs1.err

source ~/inguruneak/GRAL/bin/activate

echo "Proben hasieran"

echo "ZeroShot"

echo "Mixtral"
srun python defLLMs.py modeloak/Mixtral-8x7B-Instruct-v0.1 ModelsOutputs/ZeroShot/Mixtral.json Zero
echo "Mistral1"
srun python defLLMs.py modeloak/Mistral-7B-Instruct-v0.1 ModelsOutputs/ZeroShot/Mistral1.json Zero
echo "Mistral2"
srun python defLLMs.py modeloak/Mistral-7B-Instruct-v0.2 ModelsOutputs/ZeroShot/Mistral2.json Zero
echo "ZephyrA"
srun python defLLMs.py modeloak/zephyr-7b-alpha ModelsOutputs/ZeroShot/ZephyrA.json Zero
echo "ZephyrB"
srun python defLLMs.py modeloak/zephyr-7b-beta ModelsOutputs/ZeroShot/ZephyrB.json Zero


echo "FewShot"

echo "Mixtral"
srun python defLLMs.py modeloak/Mixtral-8x7B-Instruct-v0.1 ModelsOutputs/FewShot/Mixtral.json Few
echo "Mistral1"
srun python defLLMs.py modeloak/Mistral-7B-Instruct-v0.1 ModelsOutputs/FewShot/Mistral1.json Few
echo "Mistral2"
srun python defLLMs.py modeloak/Mistral-7B-Instruct-v0.2 ModelsOutputs/FewShot/Mistral2.json Few
echo "ZephyrA"
srun python defLLMs.py modeloak/zephyr-7b-alpha ModelsOutputs/FewShot/ZephyrA.json Few
echo "ZephyrB"
srun python defLLMs.py modeloak/zephyr-7b-beta ModelsOutputs/FewShot/ZephyrB.json Few

echo "Emaitzak kalkulatzen"
srun python applyOtherScores.py
srun python AvgStdModels.py ModelsOutputs/ZeroShot ModelsOutputs/ZeroShot/results.csv
srun python AvgStdModels.py ModelsOutputs/FewShot ModelsOutputs/FewShot/results.csv
echo "Amaiera"