#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-eeg_fmri_fusion_variants-llm_modeling_nsd-02_plot
#SBATCH --mail-type=end
#SBATCH --mem=7000
#SBATCH --time=02:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a eeg_reps_all
declare -a regression_all
index=0
for rep in 'average' ; do
    for reg in 'linear' 'ridge' ; do
        eeg_reps_all[$index]=$rep
        regression_all[$index]=$reg
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
eeg_reps=${eeg_reps_all[$SLURM_ARRAY_TASK_ID]}
regression=${regression_all[$SLURM_ARRAY_TASK_ID]}
echo eeg_reps: $eeg_reps
echo regression: $regression

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion_variants/llm_modeling_nsd

# Run the job
python 02_plot.py --eeg_reps $eeg_reps --regression $regression