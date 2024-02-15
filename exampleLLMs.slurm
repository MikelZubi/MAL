#!/bin/bash
#SBATCH --job-name=Example_LLMs
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --gres=gpu:2
#SBATCH --output=./SLURM/LOG/irteeraExample_LLMs1.log
#SBATCH --error=./SLURM/ERRORE/erroreExample_LLMs1.err

source ~/inguruneak/GRAL/bin/activate

echo "Proben hasieran"

echo "ZeroShotExample"

echo "Mixtral"
srun python exampleLLMs.py modeloak/Mixtral-8x7B-Instruct-v0.1 ModelsOutputs/ZeroShotExample/Mixtral.json Zero
echo "Mistral2"
srun python exampleLLMs.py modeloak/Mistral-7B-Instruct-v0.2 ModelsOutputs/ZeroShotExample/Mistral2.json Zero
echo "ZephyrB"
srun python exampleLLMs.py modeloak/zephyr-7b-beta ModelsOutputs/ZeroShotExample/ZephyrB.json Zero


echo "FewShotExample"

echo "Mixtral"
srun python exampleLLMs.py modeloak/Mixtral-8x7B-Instruct-v0.1 ModelsOutputs/FewShotExample/Mixtral.json Few
echo "Mistral2"
srun python exampleLLMs.py modeloak/Mistral-7B-Instruct-v0.2 ModelsOutputs/FewShotExample/Mistral2.json Few
echo "ZephyrB"
srun python exampleLLMs.py modeloak/zephyr-7b-beta ModelsOutputs/FewShotExample/ZephyrB.json Few

echo "Emaitzak kalkulatzen"
srun python AvgStdModels.py ModelsOutputs/ZeroShotExample ModelsOutputs/ZeroShotExample/results.csv
srun python AvgStdModels.py ModelsOutputs/FewShotExample ModelsOutputs/FewShotExample/results.csv
echo "Amaiera"