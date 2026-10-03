#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-03_relationship_prediction_explanation-fmri-02_plot
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:05:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/03_relationship_prediction_explanation/fmri

# Run the job
python 02_plot.py