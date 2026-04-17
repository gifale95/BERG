#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-retinotopy-04_plot
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=01:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a encoding_model_all
declare -a subjects_all
declare -a GRID_RES_all
declare -a PROBE_SIGMA_all
declare -a BG_VALUE_all
index=0
for em in 'fmri-nsd_fsaverage-huze' 'fmri-nsd_fsaverage-alexnet' 'fmri-nsd_fsaverage-alexnet_untrained' ; do
    for sub in `seq 1 1` ; do
        for g in '40' '60' '80' '100' ; do
            for s in '0.25' '0.5' '0.75' ; do
                for b in '0.5' ; do
                    encoding_model_all[$index]=$em
                    subjects_all[$index]=$sub
                    GRID_RES_all[$index]=$g
                    PROBE_SIGMA_all[$index]=$s
                    BG_VALUE_all[$index]=$b
                    ((index=index+1))
                done
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
encoding_model=${encoding_model_all[$SLURM_ARRAY_TASK_ID]}
subject=${subjects_all[$SLURM_ARRAY_TASK_ID]}
GRID_RES=${GRID_RES_all[$SLURM_ARRAY_TASK_ID]}
PROBE_SIGMA=${PROBE_SIGMA_all[$SLURM_ARRAY_TASK_ID]}
BG_VALUE=${BG_VALUE_all[$SLURM_ARRAY_TASK_ID]}
echo encoding_model: $encoding_model
echo SUBJECT: $subject
echo GRID_RES: $GRID_RES
echo PROBE_SIGMA: $PROBE_SIGMA
echo BG_VALUE: $BG_VALUE

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/fmri/retinotopy

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python 04_plot.py --encoding_model $encoding_model --subject $subject --GRID_RES $GRID_RES --PROBE_SIGMA $PROBE_SIGMA --BG_VALUE $BG_VALUE