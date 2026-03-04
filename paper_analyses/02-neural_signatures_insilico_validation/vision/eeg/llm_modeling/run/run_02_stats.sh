#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-eeg-llm_modeling-02_stats
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:20:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a channels_all
declare -a encoding_model_all
index=0
for c in 'O,P' ; do
    for em in 'eeg-things_eeg_2-alexnet' 'eeg-things_eeg_2-alexnet_untrained' ; do
        channels_all[$index]=$c
        encoding_model_all[$index]=$em
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
channels=${channels_all[$SLURM_ARRAY_TASK_ID]}
echo channels: $channels
encoding_model=${encoding_model_all[$SLURM_ARRAY_TASK_ID]}
echo encoding_model: $encoding_model

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/eeg/llm_modeling

# Run the job
python 02_stats.py --channels $channels --encoding_model $encoding_model