#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=ccn_2026_tutorial-01_extract_images_coco
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=06:00:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/ccn_2026_tutorial

# Run the job
python 01_extract_images_coco.py