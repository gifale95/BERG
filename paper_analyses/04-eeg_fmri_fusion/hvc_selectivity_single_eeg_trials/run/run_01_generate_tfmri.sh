#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_eeg_fmri_fusion-hvc_selectivity_single_eeg_trials-01_generate_tfmri
#SBATCH --mail-type=end
#SBATCH --mem=18000
#SBATCH --time=08:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a fmri_subject_all
declare -a hemisphere_all
index=0
for s in `seq 1 8` ; do
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

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/hvc_selectivity_single_eeg_trials

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 01_generate_tfmri.py --fmri_subject $fmri_subject --hemisphere $hemisphere