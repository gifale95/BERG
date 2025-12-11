#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-fmri-behavioral_modeling-00_compute_geodesic_vertex_distances
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=10:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a hemisphere_all
declare -a vertex_split_all
index=0
for h in 'lh' ; do
    for v in '2' '3' ; do
        hemisphere_all[$index]=$h
        vertex_split_all[$index]=$v
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
vertex_split=${vertex_split_all[$SLURM_ARRAY_TASK_ID]}
echo hemisphere: $hemisphere
echo vertex_split: $vertex_split

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/fmri/behavioral_modeling

# Run the job
python 00_compute_geodesic_vertex_distances.py --hemisphere $hemisphere --vertex_split $vertex_split