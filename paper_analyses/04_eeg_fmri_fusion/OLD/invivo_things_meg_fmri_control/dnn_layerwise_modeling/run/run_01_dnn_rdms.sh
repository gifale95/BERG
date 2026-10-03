#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_eeg_fmri_fusion-invivo_things_meg_fmri_control-dnn_layerwise_modeling-01_dnn_rdms
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=01:00:00
#SBATCH --qos=hiprio
#SBATCH --partition=agcichy
#SBATCH --gres=gpu:1 # number of GPUs

# CUDA module
module add CUDA/12.4.0

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
python 01_dnn_rdms.py --dnn_model $dnn_model