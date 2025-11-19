#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-ffa_ppa-01_generate_insilico_fmri
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=01:00:00
#SBATCH --qos=hiprio
#SBATCH --partition=agcichy
#SBATCH --gres=gpu:1 # number of GPUs

# CUDA module
module add CUDA/12.4.0

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/neural_signatures_insilico_validation/vision/fmri/ffa_ppa

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 01_generate_insilico_fmri.py