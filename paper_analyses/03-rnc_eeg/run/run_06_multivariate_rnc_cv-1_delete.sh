#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=03-rnc_eeg-06_multivariate_rnc_cv-1
#SBATCH --mail-type=end
#SBATCH --mem=2500
#SBATCH --time=05:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a cv_subject_all
declare -a time_pair_all
declare -a control_condition_all
index=0
for s in '1' ; do
    for t in '0.1-0.4' ; do
        for c in 'disentangle' ; do
            cv_subject_all[$index]=$s
            time_pair_all[$index]=$t
            control_condition_all[$index]=$c
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
cv_subject=${cv_subject_all[$SLURM_ARRAY_TASK_ID]}
time_pair=${time_pair_all[$SLURM_ARRAY_TASK_ID]}
control_condition=${control_condition_all[$SLURM_ARRAY_TASK_ID]}
echo cv_subject: $cv_subject
echo time_pair: $time_pair
echo control_condition: $control_condition

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/03-rnc_eeg

# Run the job
python 06_multivariate_rnc.py --cv_subject $cv_subject --time_pair $time_pair --control_condition $control_condition --cv '1'
