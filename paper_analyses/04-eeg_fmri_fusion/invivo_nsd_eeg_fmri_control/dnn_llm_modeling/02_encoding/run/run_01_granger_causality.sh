#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-invivo_nsd_eeg_fmri_control-granger_causality-encoding-01_granger_causality
#SBATCH --mail-type=end
#SBATCH --mem=300000
#SBATCH --time=40:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a eeg_train_trials_all
declare -a regression_all
index=0
for fs in '1' ; do # !!! '1' '4' '5' '6' '7' '8'
    for t in 'even' ; do # !!! 'even' 'odd'
        for r in 'linear' 'ridge' ; do
            subject_all[$index]=$fs
            eeg_train_trials_all[$index]=$t
            regression_all[$index]=$r
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
eeg_train_trials=${eeg_train_trials_all[$SLURM_ARRAY_TASK_ID]}
regression=${regression_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo eeg_train_trials: $eeg_train_trials
echo regression: $regression

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/invivo_nsd_eeg_fmri_control/granger_causality/encoding

# Run the job
python 01_granger_causality.py --subject $subject --eeg_train_trials $eeg_train_trials --regression $regression