#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-fmri-behavioral_modeling-00b_merge_geodesic_vertex_distances
#SBATCH --mail-type=end
#SBATCH --mem=250000
#SBATCH --time=01:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a hemisphere_all
index=0
for h in 'lh' 'rh' ; do
    hemisphere_all[$index]=$h
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
echo hemisphere: $hemisphere

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/fmri/behavioral_modeling

# Run the job
python 00b_merge_geodesic_vertex_distances.py --hemisphere $hemisphere