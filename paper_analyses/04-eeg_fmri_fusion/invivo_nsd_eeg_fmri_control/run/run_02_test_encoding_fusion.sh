#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-invivo_nsd_eeg_fmri_control-01_test_encoding_fusion
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=02:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a hemisphere_all
index=0
for s in '1' '2' '5' '7' ; do
    for h in 'lh' 'rh' ; do
        subject_all[$index]=$s
        hemisphere_all[$index]=$h
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo hemisphere: $hemisphere

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/invivo_nsd_eeg_fmri_control

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python 02_test_encoding_fusion.py --subject $subject --hemisphere $hemisphere