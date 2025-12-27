#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=psn_denoising-behavioral_modeling-02_stats
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:20:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a psn_mode_all
index=0
for p in '1' '2' ; do
    psn_mode_all[$index]=$p
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
psn_mode=${psn_mode_all[$SLURM_ARRAY_TASK_ID]}
echo psn_mode: $psn_mode

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/00_psn_denoising/eeg/behavioral_modeling

# Run the job
python 02_stats.py --psn_mode $psn_mode