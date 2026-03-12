#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-neural_control-encoding_accuracy-01_generate_insilico_responses
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=01:00:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/ale/Downloads/BERG_GITHUB/BERG/paper_analyses/05_neural_control/encoding_accuracy

# Run the job
python 01_generate_insilico_responses.py