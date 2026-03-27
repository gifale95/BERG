#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-fmri-encoding_accuracy_and_noise_analysis-04_plot
#SBATCH --mail-type=end
#SBATCH --mem=15000
#SBATCH --time=02:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a threshold_all
index=0
for t in '0' '1'; do
    threshold_all[$index]=$t
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
threshold=${threshold_all[$SLURM_ARRAY_TASK_ID]}
echo threshold: $threshold

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/fmri/encoding_accuracy_and_noise_analysis

# Run the job
python 04_plot.py --threshold $threshold