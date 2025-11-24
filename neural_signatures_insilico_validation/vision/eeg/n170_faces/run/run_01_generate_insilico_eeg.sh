#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-eeg_n170_faces-01_generate_insilico_eeg
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:15:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/neural_signatures_insilico_validation/vision/eeg/n170_faces

# Run the job
python 01_generate_insilico_eeg.py