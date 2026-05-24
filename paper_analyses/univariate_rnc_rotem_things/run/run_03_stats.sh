#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=univariate_rnc_rotem_things-03_stats
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:10:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a cv_all
index=0
for c in '0' '1' ; do
    cv_all[$index]=$c
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
cv=${cv_all[$SLURM_ARRAY_TASK_ID]}
echo cv: $cv

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/univariate_rnc_rotem_things

# Run the job
python 03_stats.py --cv $cv