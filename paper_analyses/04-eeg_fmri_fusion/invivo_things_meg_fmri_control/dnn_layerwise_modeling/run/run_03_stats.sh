#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_eeg_fmri_fusion-invivo_things_meg_fmri_control-dnn_layerwise_modeling-03_stats
#SBATCH --mail-type=end
#SBATCH --mem=50000
#SBATCH --time=00:20:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a dnn_model_all
index=0
for m in 'alexnet' ; do
    dnn_model_all[$index]=$m
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
dnn_model=${dnn_model_all[$SLURM_ARRAY_TASK_ID]}
echo dnn_model: $dnn_model

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/invivo_things_meg_fmri_control/dnn_layerwise_modeling

# Run the job
python 03_stats.py --dnn_model $dnn_model