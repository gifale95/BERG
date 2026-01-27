#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-object_categorization_within_category_exemplar_decoding-03_stats
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=02:00:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/eeg/object_categorization_within_category_exemplar_decoding

# Run the job
python 03_stats.py