#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=03-rnc_eeg-01_generate_insilico_eeg
#SBATCH --mail-type=end
#SBATCH --mem=6000
#SBATCH --time=01:00:00
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
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/03-rnc_eeg

# Run the job
python 01_generate_insilico_eeg.py --subject $subject