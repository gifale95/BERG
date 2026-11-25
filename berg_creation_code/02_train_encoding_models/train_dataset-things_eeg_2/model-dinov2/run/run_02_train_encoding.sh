#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=things_eeg_2_dinov2_model-01_extract_image_features
#SBATCH --mail-type=end
#SBATCH --mem=90000
#SBATCH --time=02:00:00
#SBATCH --qos=hiprio
#SBATCH --partition=agcichy
#SBATCH --nodelist=g017
#SBATCH --gres=gpu:1 # number of GPUs

# CUDA module
module add CUDA/12.4.0

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02_insilico_capture_of_neural_signatures/fmri/hvc_selectivity

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python 01_extract_image_features.py