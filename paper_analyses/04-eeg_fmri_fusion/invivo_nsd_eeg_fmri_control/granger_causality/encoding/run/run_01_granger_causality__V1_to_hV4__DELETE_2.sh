#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-invivo_nsd_eeg_fmri_control-granger_causality-encoding-01_granger_causality__V1_to_hV4__DELETE_2
#SBATCH --mail-type=end
#SBATCH --mem=25000
#SBATCH --time=60:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a roi_source_all
declare -a roi_target_all
declare -a eeg_train_trials_all
declare -a regression_all
declare -a time_split_all
index=0
for fs in '5' ; do
    for rs in 'V1' ; do
        for rt in 'hV4' ; do
            for tr in 'even' 'odd' ; do
                for r in 'ridge' ; do
                    for t in `seq 0 9` ; do
                        subject_all[$index]=$fs
                        roi_source_all[$index]=$rs
                        roi_target_all[$index]=$rt
                        eeg_train_trials_all[$index]=$tr
                        regression_all[$index]=$r
                        time_split_all[$index]=$t
                        ((index=index+1))
                    done
                done
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
roi_source=${roi_source_all[$SLURM_ARRAY_TASK_ID]}
roi_target=${roi_target_all[$SLURM_ARRAY_TASK_ID]}
eeg_train_trials=${eeg_train_trials_all[$SLURM_ARRAY_TASK_ID]}
regression=${regression_all[$SLURM_ARRAY_TASK_ID]}
time_split=${time_split_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo roi_source: $roi_source
echo roi_target: $roi_target
echo eeg_train_trials: $eeg_train_trials
echo regression: $regression
echo time_split: $time_split

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/invivo_nsd_eeg_fmri_control/granger_causality/encoding

# Run the job
python 01_granger_causality.py --subject $subject --roi_source $roi_source --roi_target $roi_target --eeg_train_trials $eeg_train_trials --regression $regression --time_split $time_split