#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-dnn_layerwise_modeling-02_dnn_layerwise_modeling
#SBATCH --mail-type=end
#SBATCH --mem=15000
#SBATCH --time=60:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a fmri_subject_all
declare -a hemisphere_all
declare -a eeg_train_trials_all
declare -a time_split_all
index=0
for fs in `seq 1 8` ; do
    for h in 'lh' 'rh' ; do
        for eeg_train_trials in 'even' 'odd' ; do
            for t in `seq 0 19` ; do
                fmri_subject_all[$index]=$fs
                hemisphere_all[$index]=$h
                eeg_train_trials_all[$index]=$eeg_train_trials
                time_split_all[$index]=$t
                ((index=index+1))
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
eeg_train_trials=${eeg_train_trials_all[$SLURM_ARRAY_TASK_ID]}
time_split=${time_split_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo hemisphere: $hemisphere
echo eeg_train_trials: $eeg_train_trials
echo time_split: $time_split

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/dnn_layerwise_modeling

# Run the job
python 02_dnn_layerwise_modeling.py --fmri_subject $fmri_subject --hemisphere $hemisphere --eeg_train_trials $eeg_train_trials --time_split $time_split