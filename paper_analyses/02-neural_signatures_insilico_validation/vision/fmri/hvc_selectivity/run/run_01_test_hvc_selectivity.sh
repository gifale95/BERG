#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-hvc_selectivity-01_test_hvc_selectivity
#SBATCH --mail-type=end
#SBATCH --mem=25000
#SBATCH --time=00:10:00
#SBATCH --qos=hiprio
#SBATCH --partition=agcichy
#SBATCH --gres=gpu:1 # number of GPUs

# CUDA module
module add CUDA/12.4.0

# Create the parameters combinations
declare -a encoding_model_all
declare -a subject_all
index=0
for em in 'fmri-nsd_fsaverage-huze' 'fmri-nsd_fsaverage-vit_b_32' ; do
    for s in `seq 1 8` ; do
        encoding_model_all[$index]=$em
        subject_all[$index]=$s
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
encoding_model=${encoding_model_all[$SLURM_ARRAY_TASK_ID]}
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo encoding_model: $encoding_model

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/fmri/hvc_selectivity

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 01_test_hvc_selectivity.py --subject $subject --encoding_model $encoding_model