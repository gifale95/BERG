#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-04_eeg_fmri_fusion-behavioral_modeling-03_plot
#SBATCH --mail-type=end
#SBATCH --mem=7000
#SBATCH --time=05:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a source_dataset_all
index=0
for d in 'things_meg_1' ; do
    source_dataset_all[$index]=$d
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
source_dataset=${source_dataset_all[$SLURM_ARRAY_TASK_ID]}
echo source_dataset: $source_dataset

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/behavioral_modeling

# Run the job
python 03_plot.py --source_dataset $source_dataset