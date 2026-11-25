#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=things_eeg_2_dinov2_model-02_plot
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=00:05:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a channels_all
declare -a pcs_all
index=0
for c in 'O' 'P' 'T' 'C' 'F' 'all' ; do
    for p in '100' '250' '500' '750' '1000' ; do
        channels_all[$index]=$c
        pcs_all[$index]=$p
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
channels=${channels_all[$SLURM_ARRAY_TASK_ID]}
pcs=${pcs_all[$SLURM_ARRAY_TASK_ID]}
echo channels: $channels
echo pcs: $pcs

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/berg_creation_code/03_test_encoding_models/train_dataset-things_eeg_2

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python 02_plot.py --channels $channels --pcs $pcs