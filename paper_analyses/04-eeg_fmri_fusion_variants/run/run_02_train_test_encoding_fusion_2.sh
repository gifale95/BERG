#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion_variants-02_train_test_encoding_fusion_2
#SBATCH --mail-type=end
#SBATCH --mem=50000
#SBATCH --time=50:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a fmri_subject_all
declare -a hemisphere_all
declare -a eeg_subject_all
index=0
for fs in `seq 1 2` ; do
    for h in 'lh' 'rh' ; do
        for es in `seq 1 2` ; do
            fmri_subject_all[$index]=$fs
            hemisphere_all[$index]=$h
            eeg_subject_all[$index]=$es
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
eeg_subject=${eeg_subject_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo hemisphere: $hemisphere
echo eeg_subject: $eeg_subject

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion_variants

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 02_train_test_encoding_fusion.py --fmri_subject $fmri_subject --hemisphere $hemisphere --eeg_subject $eeg_subject --eeg_reps 'single' --regression 'ridge'