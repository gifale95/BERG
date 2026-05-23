#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=univariate_rnc_rotem-00b_aggregate_images
#SBATCH --mail-type=end
#SBATCH --mem=500000
#SBATCH --time=02:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a imagenet_split_all
index=0
for i in 'train' 'val' ; do
    imagenet_split_all[$index]=$i
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
imagenet_split=${imagenet_split_all[$SLURM_ARRAY_TASK_ID]}
echo imagenet_split: $imagenet_split

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/univariate_rnc_rotem

# Run the job
python 00b_aggregate_images.py --imagenet_split $imagenet_split