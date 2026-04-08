#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-neural_control-03_neural_control_roi-pair
#SBATCH --mail-type=end
#SBATCH --mem=3500
#SBATCH --time=00:05:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a control_roi_1_all
declare -a control_roi_2_all
index=0
for s in 'N' 'F' ; do
    for c1 in 'early-drive_late-drive' 'early-suppress_late-suppress' 'early-drive_late-suppress' 'early-suppress_late-drive' ; do
        for c2 in 'early-drive_late-drive' 'early-suppress_late-suppress' 'early-drive_late-suppress' 'early-suppress_late-drive' ; do
            subject_all[$index]=$s
            control_roi_1_all[$index]=$c1
            control_roi_2_all[$index]=$c2
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
control_roi_1=${control_roi_1_all[$SLURM_ARRAY_TASK_ID]}
control_roi_2=${control_roi_2_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo control_roi_1: $control_roi_1
echo control_roi_2: $control_roi_2

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/05_neural_control/neural_control/

# Run the job
python 03_neural_control.py --subject $subject --roi_1 'V1' --roi_2 'V4' --control_roi_1 $control_roi_1 --control_roi_2 $control_roi_2