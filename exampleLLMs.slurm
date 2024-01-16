#!/bin/bash
#SBATCH --job-name=Test_LLMs
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM/LOG/irteeraExampleLLMs1.log
#SBATCH --error=./SLURM/ERRORE/erroreExampleLLMs1.err

source ~/inguruneak/GRAL/bin/activate

echo "Proben hasieran"
echo "Mistral1"
srun python exampleLLMs.py modeloak/zephyr-7b-beta ModelsOutputs/FewShotExample/ZephyrB.json Few


echo "Emaitzak kalkulatzen"
srun python AvgStdModels.py ModelsOutputs/FewShotExample ModelsOutputs/ZeroShotExample/results.csv
echo "Amaiera"
