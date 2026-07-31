#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-retinotopy-04_plot
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=04:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a fmri_subject_all
index=0
for s in `seq 1 8` ; do
    fmri_subject_all[$index]=$s
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/retinotopy

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 04_plot.py --fmri_subject $fmri_subject