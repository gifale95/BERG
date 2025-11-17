#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-retinotopy-01_create_stimuli
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=03:00:00
#SBATCH --qos=prio

# Create the parameters combinations
declare -a GRID_RES_all
declare -a PROBE_SIGMA_all
declare -a BG_VALUE_all
index=0
for g in '40' ; do
    for s in '0.25' '0.5' '0.75' '1' ; do
        for b in '0.5' ; do
            GRID_RES_all[$index]=$g
            PROBE_SIGMA_all[$index]=$s
            BG_VALUE_all[$index]=$b
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
GRID_RES=${GRID_RES_all[$SLURM_ARRAY_TASK_ID]}
PROBE_SIGMA=${PROBE_SIGMA_all[$SLURM_ARRAY_TASK_ID]}
BG_VALUE=${BG_VALUE_all[$SLURM_ARRAY_TASK_ID]}
echo GRID_RES: $GRID_RES
echo PROBE_SIGMA: $PROBE_SIGMA
echo BG_VALUE: $BG_VALUE

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/neural_signatures_insilico_validation/vision/fmri/retinotopy

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 01_create_stimuli.py --GRID_RES $GRID_RES --PROBE_SIGMA $PROBE_SIGMA --BG_VALUE $BG_VALUE