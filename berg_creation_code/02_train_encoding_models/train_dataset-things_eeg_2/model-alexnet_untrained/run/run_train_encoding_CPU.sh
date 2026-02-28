#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-things_eeg_2_alexnet_untrained-train_encoding_CPU
#SBATCH --mail-type=end
#SBATCH --mem=70000
#SBATCH --time=20:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
index=0
for s in `seq 1 10` ; do
    subject_all[$index]=$s
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/berg_creation_code/02_train_encoding_models/train_dataset-things_eeg_2/model-alexnet_untrained

# Run the job
python train_encoding.py --subject $subject