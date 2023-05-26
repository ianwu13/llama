#!/bin/bash
#SBATCH --account=glucas_540
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

# Change to proper directory
cd ../../..

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate llama

# run
echo ""
echo "*-*-ID-*-* TRAINING GCN AGG W 1 LAYERS, 2 EDGES PER NODE ON Camelyon16"
echo ""

# TCGA-lung-default
# Camelyon16
python3 train_gcn_agg.py --dataset=Camelyon16 --num_epochs=100 --model=graph_dsmil --edges_per_node=2 --gcn_layer_type=GraphConv --n_gcn_layers=1 --agg_type=dsmil

echo "DONE"