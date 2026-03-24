#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-eeg-dnn_layerwise_modeling-03_rsa
#SBATCH --mail-type=end
#SBATCH --mem=200
#SBATCH --time=00:15:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a channels_all
declare -a dnn_model_all
declare -a encoding_model_all
index=0
for s in `seq 1 10` ; do
    for c in 'O,P' ; do
        for m in 'alexnet' 'resnet50' ; do
            for em in 'eeg-things_eeg_2-vit_b_32' 'eeg-things_eeg_2-alexnet' 'eeg-things_eeg_2-alexnet_untrained' ; do
                subject_all[$index]=$s
                channels_all[$index]=$c
                dnn_model_all[$index]=$m
                encoding_model_all[$index]=$em
                ((index=index+1))
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
channels=${channels_all[$SLURM_ARRAY_TASK_ID]}
dnn_model=${dnn_model_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo channels: $channels
echo dnn_model: $dnn_model
encoding_model=${encoding_model_all[$SLURM_ARRAY_TASK_ID]}
echo encoding_model: $encoding_model

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/eeg/dnn_layerwise_modeling

# Run the job
python 03_rsa.py --subject $subject --channels $channels --dnn_model $dnn_model --encoding_model $encoding_model