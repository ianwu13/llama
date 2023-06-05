#!/bin/bash
#SBATCH --account=glucas_540
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=v100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00

TARGET_FOLDER="weights"
MODEL_SIZE="30BB"

declare -A MP_DICT
MP_DICT["7B"]="1"
MP_DICT["13B"]="2"
MP_DICT["30B"]="4"
MP_DICT["65B"]="8"

# Change to proper directory
cd ..

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate llama

# run
echo ""
echo "Running test predictions with model size ${MODEL_SIZE}"
echo ""

torchrun --nproc_per_node ${MP_DICT[$MODEL_SIZE]} example.py --ckpt_dir $TARGET_FOLDER/$MODEL_SIZE --tokenizer_path $TARGET_FOLDER/tokenizer.model

echo "DONE"