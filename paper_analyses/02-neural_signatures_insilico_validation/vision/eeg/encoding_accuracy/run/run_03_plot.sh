#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-eeg-encoding_accuracy-03_plot
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=00:10:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a encoding_model_all
index=0
for em in 'eeg-things_eeg_2-alexnet' 'eeg-things_eeg_2-alexnet_untrained' 'eeg-things_eeg_2-vit_b_32' ; do
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
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/eeg/encoding_accuracy

# Run the job
python 03_plot.py --encoding_model $encoding_model