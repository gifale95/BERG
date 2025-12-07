#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=03-rnc_eeg-06_multivariate_rnc_cv-0
#SBATCH --mail-type=end
#SBATCH --mem=2500
#SBATCH --time=04:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a time_pair_all
declare -a control_condition_all
index=0
for t in '0.1-0.2' '0.1-0.3' '0.1-0.4' '0.2-0.3' '0.2-0.4' '0.3-0.4' ; do
    for c in 'align' 'disentangle' ; do
        time_pair_all[$index]=$t
        control_condition_all[$index]=$c
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
time_pair=${time_pair_all[$SLURM_ARRAY_TASK_ID]}
control_condition=${control_condition_all[$SLURM_ARRAY_TASK_ID]}
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
python 06_multivariate_rnc.py --time_pair $time_pair --control_condition $control_condition --cv '0'