#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-neural_control-encoding_accuracy-02_encoding_accuracy
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=10:00:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/05_neural_control/encoding_accuracy

# Run the job
python 02_encoding_accuracy.py