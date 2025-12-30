#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-eeg-dnn_layerwise_modeling-03_rsa
#SBATCH --mail-type=end
#SBATCH --mem=200
#SBATCH --time=00:05:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a channels_all
declare -a model_all
index=0
for s in `seq 1 10` ; do
    for c in 'O,P' ; do
        for m in 'alexnet' 'resnet50' ; do
            subject_all[$index]=$s
            channels_all[$index]=$c
            model_all[$index]=$m
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
channels=${channels_all[$SLURM_ARRAY_TASK_ID]}
model=${model_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo channels: $channels
echo model: $model

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/eeg/dnn_layerwise_modeling

# Run the job
python 03_rsa.py --subject $subject --channels $channels --model $model