#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=univariate_rnc_rotem_things-00a_extract_images
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=05:00:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/univariate_rnc_rotem_things

# Run the job
python 00a_extract_images.py