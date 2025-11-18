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
declare -a imagesall
index=0
for i in 'texforms' ; do
    images_all[$index]=$i
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
images=${images_all[$SLURM_ARRAY_TASK_ID]}
echo images: $images

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/neural_signatures_insilico_validation/vision/fmri/tripartite_organization

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 01_generate_insilico_fmri.py --images $images