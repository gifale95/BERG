#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=univariate_rnc_rotem-00a_extract_images__imagenet_split-train
#SBATCH --mail-type=end
#SBATCH --mem=5500
#SBATCH --time=05:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a current_batch_all
index=0
for b in `seq 0 99` ; do
    current_batch_all[$index]=$b
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
current_batch=${current_batch_all[$SLURM_ARRAY_TASK_ID]}
echo current_batch: $current_batch

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/univariate_rnc_rotem

# Run the job
python 00a_extract_images.py --current_batch $current_batch --imagenet_split 'train'