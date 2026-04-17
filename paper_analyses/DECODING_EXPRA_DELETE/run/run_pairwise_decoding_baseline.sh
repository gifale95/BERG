#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=EXPRA-01_pairwise_decoding_eeg-baseline
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=30:00:00
#SBATCH --qos=extended

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/DECODING_EXPRA_DELETE

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python 01_pairwise_decoding_eeg.py --n_conditions '200' --n_repeats '80' --cv '1'