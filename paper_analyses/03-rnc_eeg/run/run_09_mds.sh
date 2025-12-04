#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=03-rnc_eeg-09_mds
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=20:00:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/03-rnc_eeg

# Run the job
python 09_mds.py