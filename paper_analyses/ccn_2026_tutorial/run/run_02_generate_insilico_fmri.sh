#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=ccn_2026_tutorial-02_generate_insilico_fmri
#SBATCH --mail-type=end
#SBATCH --mem=25000
#SBATCH --time=01:00:00
#SBATCH --qos=extended
#SBATCH --partition=agcichy
#SBATCH --gres=gpu:1 # number of GPUs

# CUDA module
module add CUDA/12.4.0

# Create the parameters combinations
declare -a fmri_subject_all
declare -a image_set_all
index=0
for fs in '1' '2' '3' '4' '5' '6' '7' '8' ; do
    for i in 'imagenet' 'coco' ; do
        fmri_subject_all[$index]=$fs
        image_set_all[$index]=$i
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
image_set=${image_set_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo image_set: $image_set

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/ccn_2026_tutorial

# Run the job
python 02_generate_insilico_fmri.py --fmri_subject $fmri_subject --image_set $image_set