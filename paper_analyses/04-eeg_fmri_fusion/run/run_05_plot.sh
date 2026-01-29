#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-05_plot
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=02:00:00
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

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python 05_plot.py --source_dataset $source_dataset