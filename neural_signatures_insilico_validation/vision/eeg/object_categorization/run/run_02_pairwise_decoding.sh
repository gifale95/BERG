#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-object_categorization-02_pairwise_decoding
#SBATCH --mail-type=end
#SBATCH --mem=3000
#SBATCH --time=10:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
index=0
for sub in `seq 1 10` ; do
    subject_all[$index]=$sub
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
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/neural_signatures_insilico_validation/vision/eeg/object_categorization

# Run the job
python 02_pairwise_decoding.py --subject $subject