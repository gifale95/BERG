#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-erps-01_erps
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:30:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a encoding_model_all
index=0
for em in 'eeg-things_eeg_2-alexnet' 'eeg-things_eeg_2-alexnet_untrained' ; do
    encoding_model_all[$index]=$em
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
encoding_model=${encoding_model_all[$SLURM_ARRAY_TASK_ID]}
echo encoding_model: $encoding_model

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/eeg/erps

# Run the job
python 01_erps.py --encoding_model $encoding_model