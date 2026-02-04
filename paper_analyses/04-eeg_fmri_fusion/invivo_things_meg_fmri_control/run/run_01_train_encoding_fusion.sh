#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-invivo_things_meg_fmri_control-01_train_encoding_fusion
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=02:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a fmri_subject_all
index=0
for fs in `seq 1 3` ; do
    fmri_subject_all[$index]=$fs
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/invivo_things_meg_fmri_control

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python 01_train_encoding_fusion.py --fmri_subject $fmri_subject