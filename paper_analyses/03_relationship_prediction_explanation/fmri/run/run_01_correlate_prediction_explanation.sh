#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-03_relationship_prediction_explanation-fmri-01_correlate_prediction_explanation
#SBATCH --mail-type=end
#SBATCH --mem=3000
#SBATCH --time=00:10:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/03_relationship_prediction_explanation/fmri

# Run the job
python 01_correlate_prediction_explanation.py