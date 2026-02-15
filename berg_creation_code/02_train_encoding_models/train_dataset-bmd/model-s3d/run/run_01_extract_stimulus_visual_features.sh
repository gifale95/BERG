#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-bmd_model-train_encoding-01_extract_stimulus_visual_features
#SBATCH --mail-type=end
#SBATCH --mem=90000
#SBATCH --time=10:00:00
#SBATCH --qos=extended
#SBATCH --partition=agcichy
#SBATCH --gres=gpu:1 # number of GPUs

# CUDA module
module add CUDA/12.4.0

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/berg_creation_code/02_train_encoding_models/train_dataset-bmd/model-s3d

# Run the job
python 01_extract_stimulus_visual_features.py