#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-fmri-dnn_layerwise_modeling_100_voxel_neighbor-04_stats
#SBATCH --mail-type=end
#SBATCH --mem=3000
#SBATCH --time=00:10:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a model_all
declare -a encoding_model_all
index=0
for m in 'alexnet' 'resnet50' ; do
    for em in 'fmri-nsd_fsaverage-huze' ; do
        model_all[$index]=$m
        encoding_model_all[$index]=$em
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
model=${model_all[$SLURM_ARRAY_TASK_ID]}
encoding_model=${encoding_model_all[$SLURM_ARRAY_TASK_ID]}
echo model: $model
echo encoding_model: $encoding_model

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/fmri/dnn_layerwise_modeling_100_voxel_neighbor

# Run the job
python 04_stats.py --model $model --encoding_model $encoding_model