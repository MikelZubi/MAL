#!/bin/bash
#SBATCH --job-name=Def_LLMs
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:2
#SBATCH --output=./SLURM/LOG/irteeraTest_LLMs1.log
#SBATCH --error=./SLURM/ERRORE/erroreTest_LLMs1.err

source ~/inguruneak/GRAL/bin/activate

echo "Proben hasieran"

echo "ZeroShot"


echo "Mixtral"
srun python defLLMs.py modeloak/Mixtral-8x7B-Instruct-v0.1 ModelsOutputs/ZeroShot/Mixtral.json Mistral 0
echo "Mistral2"
srun python defLLMs.py modeloak/Mistral-7B-Instruct-v0.2 ModelsOutputs/ZeroShot/Mistral2.json Mistral 0
echo "ZephyrB"
srun python defLLMs.py modeloak/zephyr-7b-beta ModelsOutputs/ZeroShot/ZephyrB.json Zephyr 0



echo "FewShot"

echo "Mixtral"
srun python defLLMs.py modeloak/Mixtral-8x7B-Instruct-v0.1 ModelsOutputs/FewShot/Mixtral.json Mistral 3
echo "Mistral2"
srun python defLLMs.py modeloak/Mistral-7B-Instruct-v0.2 ModelsOutputs/FewShot/Mistral2.json  Mistral 3
echo "ZephyrB"
srun python defLLMs.py modeloak/zephyr-7b-beta ModelsOutputs/FewShot/ZephyrB.json Zephyr 3



echo "Emaitzak kalkulatzen"
srun python applyOtherScores.py
srun python AvgStdModels.py ModelsOutputs/ZeroShot ModelsOutputs/ZeroShot/results.csv
srun python AvgStdModels.py ModelsOutputs/FewShot ModelsOutputs/FewShot/results.csv
echo "Amaiera"