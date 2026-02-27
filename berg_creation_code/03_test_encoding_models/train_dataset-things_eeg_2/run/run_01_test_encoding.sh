#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-things_eeg_2_alexnet-01_test_encoding
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=01:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a model_all
index=0
for s in `seq 1 10` ; do
    for m in 'alexnet_untrained' 'alexnet' ; do
        subject_all[$index]=$s
        model_all[$index]=$m
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
model=${model_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo model: $model

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/berg_creation_code/03_test_encoding_models/train_dataset-things_eeg_2

# Run the job
python 01_test_encoding.py --subject $subject --model $model