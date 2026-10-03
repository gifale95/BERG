#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion_variants-01b_prepare_invivo_eeg
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=00:05:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a eeg_subject_all
declare -a eeg_reps_all
declare -a regression_all
index=0
for s in `seq 1 10` ; do
    for rep in 'single' 'average' ; do
        for reg in 'linear' 'ridge' ; do
            eeg_subject_all[$index]=$s
            eeg_reps_all[$index]=$rep
            regression_all[$index]=$reg
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
eeg_subject=${eeg_subject_all[$SLURM_ARRAY_TASK_ID]}
eeg_reps=${eeg_reps_all[$SLURM_ARRAY_TASK_ID]}
regression=${regression_all[$SLURM_ARRAY_TASK_ID]}
echo eeg_subject: $eeg_subject
echo eeg_reps: $eeg_reps
echo regression: $regression

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion_variants

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 01b_prepare_invivo_eeg.py --eeg_subject $eeg_subject --eeg_reps $eeg_reps --regression $regression
