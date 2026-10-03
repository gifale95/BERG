#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion_ridge-01b_prepare_invivo_eeg
#SBATCH --mail-type=end
#SBATCH --mem=60000
#SBATCH --time=02:00:00
#SBATCH --qos=extended

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion_ridge

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 01b_prepare_invivo_eeg.py