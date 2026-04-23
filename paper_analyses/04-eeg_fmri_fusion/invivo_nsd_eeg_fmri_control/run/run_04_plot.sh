#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-invivo_things_meg_fmri_control-04_plot
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:10:00
#SBATCH --qos=extended

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/invivo_things_meg_fmri_control

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python 04_plot.py