#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_eeg_fmri_fusion-dnn_layerwise_modeling-02_rsa
#SBATCH --mail-type=end
#SBATCH --mem=30000
#SBATCH --time=20:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a fmri_subject_all
declare -a hemisphere_all
declare -a dnn_model_all
index=0
for s in `seq 1 8` ; do
    for h in 'lh' 'rh' ; do
        for m in 'alexnet' 'resnet50' ; do
            fmri_subject_all[$index]=$s
            hemisphere_all[$index]=$h
            dnn_model_all[$index]=$m
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
dnn_model=${dnn_model_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo hemisphere: $hemisphere
echo dnn_model: $dnn_model

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/dnn_layerwise_modeling

# Run the job
python 02_rsa.py --fmri_subject $fmri_subject --hemisphere $hemisphere --dnn_model $dnn_model