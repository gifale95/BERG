#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-tripartite_organization-01_generate_insilico_fmri
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=00:15:00
#SBATCH --qos=hiprio
#SBATCH --partition=agcichy
#SBATCH --gres=gpu:1 # number of GPUs

# CUDA module
module add CUDA/12.4.0

# Create the parameters combinations
declare -a encoding_model_all
declare -a images_all
index=0
for em in 'fmri-nsd_fsaverage-huze' 'fmri-nsd_fsaverage-vit_b_32' ; do
    for i in 'naturalistic' 'texforms' ; do
        encoding_model_all[$index]=$em
        images_all[$index]=$i
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
encoding_model=${encoding_model_all[$SLURM_ARRAY_TASK_ID]}
images=${images_all[$SLURM_ARRAY_TASK_ID]}
echo encoding_model: $encoding_model
echo images: $images

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/neural_signatures_insilico_validation/vision/fmri/tripartite_organization

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 01_generate_insilico_fmri.py --images $images --encoding_model $encoding_model