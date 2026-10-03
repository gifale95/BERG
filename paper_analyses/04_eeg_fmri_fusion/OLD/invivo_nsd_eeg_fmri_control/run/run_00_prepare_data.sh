#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-invivo_nsd_eeg_fmri_control-00_prepare_data
#SBATCH --mail-type=end
#SBATCH --mem=100000
#SBATCH --time=01:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
index=0
for s in '1' '4' '5' '6' '7' '8' ; do
    subject_all[$index]=$s
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/invivo_nsd_eeg_fmri_control

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python 00_prepare_data.py --subject $subject