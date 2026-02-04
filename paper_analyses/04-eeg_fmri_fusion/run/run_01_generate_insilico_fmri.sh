#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-01a_generate_insilico_fmri
#SBATCH --mail-type=end
#SBATCH --mem=95000
#SBATCH --time=03-00:00:00
#SBATCH --qos=prio
#SBATCH --partition=agcichy
#SBATCH --gres=gpu:1 # number of GPUs

# CUDA module
module add CUDA/12.4.0

# Create the parameters combinations
declare -a fmri_subject_all
declare -a source_dataset_all
index=0
for s in '8' ; do
    for d in 'things_meg_1' ; do
        fmri_subject_all[$index]=$s
        source_dataset_all[$index]=$d
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
source_dataset=${source_dataset_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo source_dataset: $source_dataset

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python 01_generate_insilico_fmri.py --fmri_subject $fmri_subject --source_dataset $source_dataset
