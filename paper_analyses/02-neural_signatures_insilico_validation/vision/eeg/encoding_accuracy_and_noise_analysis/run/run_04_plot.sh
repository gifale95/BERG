#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-eeg-encoding_accuracy_and_noise_analysis-04_plot
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=00:20:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/eeg/encoding_accuracy_and_noise_analysis

# Run the job
python 04_plot.py