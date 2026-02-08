#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-03_test_encoding_fusion_things_meg_1
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=4-00:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a fmri_subject_all
declare -a hemisphere_all
declare -a source_dataset_all
index=0
for fs in `seq 1 8` ; do
    for h in 'lh' 'rh' ; do
        for d in 'things_meg_1' ; do
            fmri_subject_all[$index]=$fs
            hemisphere_all[$index]=$h
            source_dataset_all[$index]=$d
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
source_dataset=${source_dataset_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo hemisphere: $hemisphere
echo source_dataset: $source_dataset

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python 03_test_encoding_fusion.py --fmri_subject $fmri_subject --hemisphere $hemisphere --source_dataset $source_dataset