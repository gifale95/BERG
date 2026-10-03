#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion_ridge-02_train_test_encoding_fusion_eeg_reps-single
#SBATCH --mail-type=end
#SBATCH --mem=70000
#SBATCH --time=30:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a fmri_subject_all
declare -a hemisphere_all
index=0
for fs in `seq 1 8` ; do
    for h in 'lh' ; do
            fmri_subject_all[$index]=$fs
            hemisphere_all[$index]=$h
            ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo hemisphere: $hemisphere

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion_ridge

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 02_train_test_encoding_fusion.py --fmri_subject $fmri_subject --hemisphere $hemisphere $eeg_subject --eeg_reps 'single'