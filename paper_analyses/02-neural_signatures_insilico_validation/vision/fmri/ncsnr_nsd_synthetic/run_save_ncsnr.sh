#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=save_ncsnr_nsdsynthetic
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=02:00:00
#SBATCH --qos=extended

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/fmri/ncsnr_nsd_synthetic

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python save_ncsnr.py --encoding_model $encoding_model