#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-eeg-dnn_layerwise_modeling-02_eeg_rdms
#SBATCH --mail-type=end
#SBATCH --mem=3000
#SBATCH --time=04:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a channels_all
index=0
for s in `seq 1 10` ; do
    for c in 'O,P' ; do
        subject_all[$index]=$s
        channels_all[$index]=$c
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
channels=${channels_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo channels: $channels

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/eeg/dnn_layerwise_modeling

# Run the job
python 02_eeg_rdms.py --subject $subject --channels $channels