#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_creation_code-things_eeg_2-alexnet-train_encoding
#SBATCH --mail-type=end
#SBATCH --mem=200000
#SBATCH --time=02-00:00:00
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

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/berg_creation_code/02_train_encoding_models/train_dataset-things_eeg_2/model-alexnet

# Run the job
python train_encoding.py --subject $subject