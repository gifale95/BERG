#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-eeg_fmri_fusion_ridge-llm_modeling_nsd-01_rsa
#SBATCH --mail-type=end
#SBATCH --mem=50000
#SBATCH --time=40:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a fmri_subject_all
declare -a hemisphere_all
declare -a eeg_reps_all
index=0
for s in `seq 1 2` ; do
    for h in 'lh' ; do
        for rep in 'average' 'single' ; do
            fmri_subject_all[$index]=$s
            hemisphere_all[$index]=$h
            eeg_reps_all[$index]=$rep
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
eeg_reps=${eeg_reps_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo hemisphere: $hemisphere
echo eeg_reps: $eeg_reps

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion_ridge/llm_modeling_nsd

# Run the job
python 01_rsa.py --fmri_subject $fmri_subject --hemisphere $hemisphere --eeg_reps $eeg_reps