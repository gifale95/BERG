#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-tripartite_organization-02_tripartite_organization
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:20:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a encoding_model_all
declare -a images_all
index=0
for em in 'fmri-nsd_fsaverage-alexnet' 'fmri-nsd_fsaverage-alexnet_untrained' ; do
    for i in 'naturalistic' ; do
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
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/fmri/tripartite_organization

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python 02_tripartite_organization.py --encoding_model $encoding_model