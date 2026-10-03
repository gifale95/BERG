#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-01_prediction_accuracy_and_noise_analysis-fmri-04_plot
#SBATCH --mail-type=end
#SBATCH --mem=15000
#SBATCH --time=02:00:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/01_prediction_accuracy_and_noise_analysis/fmri

# Run the job
python 04_plot.py