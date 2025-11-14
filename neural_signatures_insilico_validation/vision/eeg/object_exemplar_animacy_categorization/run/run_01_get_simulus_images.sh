#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-eeg_decoding-01_get_stimulus_images
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=03:00:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/neural_signatures_insilico_validation/vision/eeg/object_exemplar_animacy_categorization

# Run the job
python 01_get_stimulus_images.py