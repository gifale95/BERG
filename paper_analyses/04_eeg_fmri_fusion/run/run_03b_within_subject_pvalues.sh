#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-03b_within_subject_pvalues
#SBATCH --mail-type=end
#SBATCH --mem=12000
#SBATCH --time=100:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a fmri_subject_all
declare -a roi_all
index=0
for fs in `seq 1 8` ; do
    for r in 'V1' 'V2' 'V3' 'hV4' 'FFA' 'PPA' ; do
        fmri_subject_all[$index]=$fs
        roi_all[$index]=$r
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo roi: $roi

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04_eeg_fmri_fusion

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python 03b_within_subject_pvalues.py --fmri_subject $fmri_subject --roi $roi