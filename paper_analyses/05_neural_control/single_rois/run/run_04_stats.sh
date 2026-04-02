#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-neural_control-04_stats
#SBATCH --mail-type=end
#SBATCH --mem=7000
#SBATCH --time=30:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a roi_all
declare -a control_all
index=0
for s in 'N' 'F' ; do
    for r in 'V1' 'V4' 'IT' ; do
        for c in 'early-drive_late-drive' 'early-suppress_late-suppress' 'early-drive_late-suppress' 'early-suppress_late-drive' ; do
            subject_all[$index]=$s
            roi_all[$index]=$r
            control_all[$index]=$c
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
control=${control_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo roi: $roi
echo control: $control

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/05_neural_control/single_rois/

# Run the job
python 04_stats.py --subject $subject --roi $roi --control $control