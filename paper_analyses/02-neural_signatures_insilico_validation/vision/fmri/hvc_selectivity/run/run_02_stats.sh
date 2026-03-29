#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-hvc_selectivity-02_stats
#SBATCH --mail-type=end
#SBATCH --mem=3000
#SBATCH --time=00:10:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a encoding_model_all
index=0
for em in 'fmri-nsd_fsaverage-alexnet' 'fmri-nsd_fsaverage-alexnet_untrained' 'fmri-nsd_fsaverage-vit_b_32' 'fmri-nsd_fsaverage-huze' ; do
    encoding_model_all[$index]=$em
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
encoding_model=${encoding_model_all[$SLURM_ARRAY_TASK_ID]}
echo encoding_model: $encoding_model

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/fmri/hvc_selectivity

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Run the job
python 02_stats.py --encoding_model $encoding_model