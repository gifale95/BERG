#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-04_eeg_fmri_fusion-invivo_things_meg_fmri_control-behavioral_modeling-02_stats
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=00:10:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/invivo_things_meg_fmri_control/behavioral_modeling

# Run the job
python 02_stats.py