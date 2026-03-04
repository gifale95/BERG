#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-fmri-dnn_layerwise_modeling-02_rsa
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=04:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a hemisphere_all
declare -a model_all
declare -a encoding_model_all
index=0
for s in `seq 1 8` ; do
    for h in 'lh' 'rh' ; do
        for m in 'alexnet' 'resnet50' ; do
            for em in 'fmri-nsd_fsaverage-alexnet' 'fmri-nsd_fsaverage-alexnet_untrained' ; do
                subject_all[$index]=$s
                hemisphere_all[$index]=$h
                model_all[$index]=$m
                encoding_model_all[$index]=$em
                ((index=index+1))
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
model=${model_all[$SLURM_ARRAY_TASK_ID]}
encoding_model=${encoding_model_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo hemisphere: $hemisphere
echo model: $model
echo encoding_model: $encoding_model

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/fmri/dnn_layerwise_modeling

# Run the job
python 02_rsa.py --subject $subject --hemisphere $hemisphere --model $model --encoding_model $encoding_model