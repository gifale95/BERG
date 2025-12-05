#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=03-rnc_eeg-05_baseline_cv-1
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=05:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a cv_subject_all
declare -a time_pair_all
index=0
for s in `seq 1 10` ; do
    for t in '0.1-0.2' '0.1-0.3' '0.1-0.4' '0.2-0.3' '0.2-0.4' '0.3-0.4' ; do
        cv_subject_all[$index]=$s
        time_pair_all[$index]=$t
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
cv_subject=${cv_subject_all[$SLURM_ARRAY_TASK_ID]}
time_pair=${time_pair_all[$SLURM_ARRAY_TASK_ID]}
echo cv_subject: $cv_subject
echo time_pair: $time_pair

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/03-rnc_eeg

# Run the job
python 05_baseline.py --cv_subject $cv_subject --time_pair $time_pair --cv '1'