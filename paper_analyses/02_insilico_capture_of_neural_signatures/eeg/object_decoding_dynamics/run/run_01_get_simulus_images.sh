#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-02_insilico_capture_of_neural_signatures-eeg-object_decoding_dynamics-01_get_stimulus_images
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:10:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02_insilico_capture_of_neural_signatures/eeg/object_decoding_dynamics

# Run the job
python 01_get_stimulus_images.py