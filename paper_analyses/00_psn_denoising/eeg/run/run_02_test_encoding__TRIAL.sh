#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=psn_denoising-eeg-02_test_encoding__TRIAL
#SBATCH --mail-type=end
#SBATCH --mem=50000
#SBATCH --time=20:00:00
#SBATCH --qos=extended

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/00_psn_denoising/eeg

# Run the job
python 02_test_encoding.py --subject $subject --psn_invivo_train '1' --psn_invivo_test '1' --psn_insilico_test '1'