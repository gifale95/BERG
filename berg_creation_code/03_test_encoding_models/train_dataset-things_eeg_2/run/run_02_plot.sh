#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_creation_code-things_eeg_2-02_plot
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=00:05:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a channels_all
declare -a model_all
index=0
for c in 'O' 'P' 'T' 'C' 'F' 'all' ; do
    for m in 'vit_b_32' 'alexnet' 'alexnet_untrained' ; do
        channels_all[$index]=$c
        model_all[$index]=$m
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
channels=${channels_all[$SLURM_ARRAY_TASK_ID]}
model=${model_all[$SLURM_ARRAY_TASK_ID]}
echo channels: $channels
echo model: $model

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/berg_creation_code/03_test_encoding_models/train_dataset-things_eeg_2

# Run the job
python 02_plot.py --channels $channels --model $model