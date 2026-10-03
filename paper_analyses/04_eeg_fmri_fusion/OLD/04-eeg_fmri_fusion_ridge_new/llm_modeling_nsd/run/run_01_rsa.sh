#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-eeg_fmri_fusion_ridge_new-llm_modeling_nsd-01_rsa
#SBATCH --mail-type=end
#SBATCH --mem=80000
#SBATCH --time=6-00:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a fmri_subject_all
declare -a hemisphere_all
index=0
for s in '1' '2' ; do
    for h in 'lh' 'rh' ; do
        fmri_subject_all[$index]=$s
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

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion_ridge_new/llm_modeling_nsd

# Run the job
python 01_rsa.py --fmri_subject $fmri_subject --hemisphere $hemisphere