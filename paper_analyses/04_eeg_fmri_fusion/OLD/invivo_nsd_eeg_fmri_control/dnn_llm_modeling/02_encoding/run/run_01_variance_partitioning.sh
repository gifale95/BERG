#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-invivo_nsd_eeg_fmri_control-dnn_llm_modeling-encoding-01_variance_partitioning
#SBATCH --mail-type=end
#SBATCH --mem=15000
#SBATCH --time=04:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a hemisphere_all
declare -a eeg_train_trials_all
declare -a time_split_all
index=0
for fs in '1' '4' '5' '6' '7' '8' ; do
    for h in 'lh' 'rh' ; do
        for tr in 'even' 'odd' ; do
            for t in `seq 0 9` ; do
                subject_all[$index]=$fs
                hemisphere_all[$index]=$h
                eeg_train_trials_all[$index]=$tr
                time_split_all[$index]=$t
                ((index=index+1))
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
eeg_train_trials=${eeg_train_trials_all[$SLURM_ARRAY_TASK_ID]}
time_split=${time_split_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo hemisphere: $hemisphere
echo eeg_train_trials: $eeg_train_trials
echo time_split: $time_split

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/invivo_nsd_eeg_fmri_control/dnn_llm_modeling/02_encoding

# Run the job
python 01_variance_partitioning.py --subject $subject --hemisphere $hemisphere --eeg_train_trials $eeg_train_trials --time_split $time_split