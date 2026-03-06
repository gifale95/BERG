#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-eeg-dnn_layerwise_modeling-05_plot
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:20:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a channels_all
declare -a dnn_model_all
declare -a encoding_model_all
index=0
for c in 'O-P' ; do
    for m in 'alexnet' 'resnet50' ; do
        for em in 'eeg-things_eeg_2-alexnet' 'eeg-things_eeg_2-alexnet_untrained' 'eeg-things_eeg_2-vit_b_32' ; do
            channels_all[$index]=$c
            dnn_model_all[$index]=$m
            encoding_model_all[$index]=$em
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
channels=${channels_all[$SLURM_ARRAY_TASK_ID]}
dnn_model=${dnn_model_all[$SLURM_ARRAY_TASK_ID]}
echo channels: $channels
echo dnn_model: $dnn_model
encoding_model=${encoding_model_all[$SLURM_ARRAY_TASK_ID]}
echo encoding_model: $encoding_model

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/eeg/dnn_layerwise_modeling

# Run the job
python 05_plot.py --channels $channels --dnn_model $dnn_model --encoding_model $encoding_model