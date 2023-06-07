#!/bin/bash
#SBATCH --job-name=llama-30b
#SBATCH --account=glucas_540
#SBATCH --partition=gpu

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:2

#SBATCH --time=00:10:00

TARGET_FOLDER="weights"
MODEL_SIZE="30B"

declare -A NNODE_DICT
NNODE_DICT["7B"]="1"
NNODE_DICT["13B"]="1"
NNODE_DICT["30B"]="2"
NNODE_DICT["65B"]="4"

declare -A MP_DICT
MP_DICT["7B"]="1"
MP_DICT["13B"]="2"
MP_DICT["30B"]="2"  # "4" total
MP_DICT["65B"]="2"  # "8" total

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

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

srun torchrun \
--nnodes ${NNODE_DICT[$MODEL_SIZE]} \
--nproc_per_node ${MP_DICT[$MODEL_SIZE]} \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
example.py --ckpt_dir $TARGET_FOLDER/$MODEL_SIZE --tokenizer_path $TARGET_FOLDER/tokenizer.model

echo "DONE"