#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-neural_control-02_neural_control
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=06:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a roi_all
declare -a control_all
index=0
for r in 'V1' 'V4' 'IT' ; do
    for c in 'drive' 'suppress' ; do
        roi_all[$index]=$r
        control_all[$index]=$c
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
control=${control_all[$SLURM_ARRAY_TASK_ID]}
echo roi: $roi
echo control: $control

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/05_neural_control

# Run the job
python 02_neural_control.py --roi $roi --control $control