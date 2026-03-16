#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-neural_control-02_neural_control
#SBATCH --mail-type=end
#SBATCH --mem=3000
#SBATCH --time=02:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a time_resolved_all
declare -a roi_all
declare -a control_all
index=0
for t in '0' '1' ; do
    for r in 'V1' 'V4' 'IT' ; do
        for c in 'drive' 'suppress' ; do
            time_resolved_all[$index]=$t
            roi_all[$index]=$r
            control_all[$index]=$c
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
time_resolved=${time_resolved_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
control=${control_all[$SLURM_ARRAY_TASK_ID]}
echo time_resolved: $time_resolved
echo roi: $roi
echo control: $control

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/05_neural_control

# Run the job
python 02_neural_control.py --time_resolved $time_resolved --roi $roi --control $control