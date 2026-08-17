#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-04_eeg_fmri_fusion-neural_dynamics-02_quantitative_dynamics_analysis-00_create_dnn_rsms
#SBATCH --mail-type=end
#SBATCH --mem=25000
#SBATCH --time=01:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a dnn_all
declare -a roi_all
declare -a time_window_pair_all
index=0
for d in 'dinov2l' ; do
    for r in 'V1' 'hV4' 'FFA' ; do
        for t in '0.05-0.10__0.20-0.25' ; do
            dnn_all[$index]=$d
            roi_all[$index]=$r
            time_window_pair_all[$index]=$t
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
dnn=${dnn_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
time_window_pair=${time_window_pair_all[$SLURM_ARRAY_TASK_ID]}
echo dnn: $dnn
echo roi: $roi
echo time_window_pair: $time_window_pair

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/neural_dynamics/02_quantitative_dynamics_analysis

# Run the job
python 00_create_dnn_rsms.py --dnn $dnn --roi $roi --time_window_pair $time_window_pair