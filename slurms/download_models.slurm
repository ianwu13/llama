#!/bin/bash
#SBATCH --account=glucas_540
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=23:00:00

# Change to proper directory
cd ..

# DON'T NEED CONDA ENV TO DOWNLOAD
# module purge
# module load conda
# eval "$(conda shell.bash hook)"
# conda activate llama

# run
echo ""
echo "Downloading LLaMA weights"
echo ""

# Run download script
source filled_download.sh

echo "DONE"