#!/bin/bash
#SBATCH --account=glucas_540
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:05:00

TARGET_FOLDER="weights"
MODEL_SIZE="7B"

declare -A MP_DICT
MP_DICT["7B"]="1"
MP_DICT["13B"]="2"

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