#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-neural_control-01_granger_causality_neural_control
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=00:10:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a rois_neural_control_all
declare -a objective_all
declare -a cv_all
index=0
for s in 'N' 'F' ; do
    for r in 'single' 'both' ; do
        for o in 'max' 'min' 'baseline' ; do
            for c in '0' '1' ; do
                subject_all[$index]=$s
                rois_neural_control_all[$index]=$r
                objective_all[$index]=$o
                cv_all[$index]=$c
                ((index=index+1))
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
rois_neural_control=${rois_neural_control_all[$SLURM_ARRAY_TASK_ID]}
objective=${objective_all[$SLURM_ARRAY_TASK_ID]}
cv=${cv_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo rois_neural_control: $rois_neural_control
echo objective: $objective
echo cv: $cv

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/05_neural_control/granger_causality

# Run the job
python 01_granger_causality_neural_control.py --subject $subject --rois_neural_control $rois_neural_control --objective $objective --cv $cv