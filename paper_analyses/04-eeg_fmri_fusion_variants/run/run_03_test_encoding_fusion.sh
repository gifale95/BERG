#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion_variants-03_test_encoding_fusion
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=00:30:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a regression_all
index=0
for r in 'linear' 'ridge' ; do
    regression_all[$index]=$r
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
regression=${regression_all[$SLURM_ARRAY_TASK_ID]}
echo regression: $regression

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion_variants

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 03_test_encoding_fusion.py --regression $regression