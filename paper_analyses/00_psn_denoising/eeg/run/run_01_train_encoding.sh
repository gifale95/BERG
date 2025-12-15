#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=psn_denoising-eeg-01_train_encoding
#SBATCH --mail-type=end
#SBATCH --mem=50000
#SBATCH --time=20:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a psn_invivo_train_all
index=0
for s in `seq 1 10` ; do
    for vtr in '0' '1' ; do
        subject_all[$index]=$s
        psn_invivo_train_all[$index]=$vtr
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
psn_invivo_train=${psn_invivo_train_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo psn_invivo_train: $psn_invivo_train

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/00_psn_denoising/eeg

# Run the job
python 01_train_encoding.py --subject $subject --psn_invivo_train $psn_invivo_train