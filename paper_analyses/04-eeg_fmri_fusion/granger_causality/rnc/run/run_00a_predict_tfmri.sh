#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-04_eeg_fmri_fusion-granger_causality-rnc-00a_predict_tfmri
#SBATCH --mail-type=end
#SBATCH --mem=75000
#SBATCH --time=30:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a fmri_subject_all
declare -a current_batch_all
index=0
for fs in `seq 1 8` ; do
    for b in `seq 0 99` ; do
        fmri_subject_all[$index]=$fs
        current_batch_all[$index]=$b
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
current_batch=${current_batch_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo current_batch: $current_batch

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/granger_causality/rnc

# Run the job
python 00a_predict_tfmri.py --fmri_subject $fmri_subject --current_batch $current_batch