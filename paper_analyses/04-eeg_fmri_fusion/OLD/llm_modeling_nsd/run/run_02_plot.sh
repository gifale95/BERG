#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-eeg_fmri_fusion-llm_modeling_nsd-02_plot
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=02:30:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/llm_modeling_nsd

# Run the job
python 02_plot.py