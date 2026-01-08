#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-eeg_fmri_fusion_variants-llm_modeling_nsd-01_rsa_regression-ridge
#SBATCH --mail-type=end
#SBATCH --mem=50000
#SBATCH --time=20:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a fmri_subject_all
declare -a hemisphere_all
declare -a eeg_subject_all
declare -a eeg_reps_all
declare -a regression_all
index=0
for s in `seq 1 2` ; do
    for h in 'lh' 'rh' ; do
        for es in `seq 1 2` ; do
            for rep in 'average' 'single' ; do
                for reg in 'ridge' ; do
                    fmri_subject_all[$index]=$s
                    hemisphere_all[$index]=$h
                    eeg_subject_all[$index]=$es
                    eeg_reps_all[$index]=$rep
                    regression_all[$index]=$reg
                    ((index=index+1))
                done
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
eeg_subject=${eeg_subject_all[$SLURM_ARRAY_TASK_ID]}
eeg_reps=${eeg_reps_all[$SLURM_ARRAY_TASK_ID]}
regression=${regression_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo hemisphere: $hemisphere
echo eeg_subject: $eeg_subject
echo eeg_reps: $eeg_reps
echo regression: $regression

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion_variants/llm_modeling_nsd

# Run the job
python 01_rsa.py --fmri_subject $fmri_subject --hemisphere $hemisphere --eeg_subject $eeg_subject --fmri_subject $fmri_subject --hemisphere $hemisphere