#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-invivo_nsd_eeg_fmri_control-03_stats
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=00:20:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a eeg_train_trials_all
index=0
for et in 'all' 'even' 'odd' ; do
    eeg_train_trials_all[$index]=$et
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
eeg_train_trials=${eeg_train_trials_all[$SLURM_ARRAY_TASK_ID]}
echo eeg_train_trials: $eeg_train_trials

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/invivo_nsd_eeg_fmri_control

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python 03_stats.py --eeg_train_trials $eeg_train_trials