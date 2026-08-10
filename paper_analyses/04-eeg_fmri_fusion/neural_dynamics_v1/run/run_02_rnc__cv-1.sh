#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-04_eeg_fmri_fusion-neural_dynamics_v1-02_rnc__cv-1
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:10:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a cv_subject_all
declare -a roi_all
declare -a time_window_pair_all
index=0
for cs in `seq 1 8` ; do
    for r in 'V1' 'hV4' 'ventral' ; do
        for t in '0.05-0.10__0.10-0.15' '0.05-0.10__0.15-0.20' '0.05-0.10__0.20-0.25' '0.10-0.15__0.15-0.20' '0.10-0.15__0.20-0.25' '0.15-0.20__0.20-0.25' ; do
            cv_subject_all[$index]=$cs
            roi_all[$index]=$r
            time_window_pair_all[$index]=$t
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
cv_subject=${cv_subject_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
time_window_pair=${time_window_pair_all[$SLURM_ARRAY_TASK_ID]}
echo cv_subject: $cv_subject
echo roi: $roi
echo time_window_pair: $time_window_pair

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/neural_dynamics_v1

# Run the job
python 02_rnc.py --cv '1' --cv_subject $cv_subject --roi $roi --time_window_pair $time_window_pair