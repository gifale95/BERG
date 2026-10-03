#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-eeg_fmri_fusion_ridge-hvc_selectivity-02_stats
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=02:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a eeg_reps_all
index=0
for rep in 'average' 'single' ; do
    eeg_reps_all[$index]=$rep
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
eeg_reps=${eeg_reps_all[$SLURM_ARRAY_TASK_ID]}
echo eeg_reps: $eeg_reps

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion_ridge/hvc_selectivity

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 02_stats.py --eeg_reps $eeg_reps