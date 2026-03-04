#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg_insilico_validation-fmri-behavioral_modeling-02_rsa
#SBATCH --mail-type=end
#SBATCH --mem=3000
#SBATCH --time=02:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a hemisphere_all
declare -a encoding_model_all
index=0
for s in `seq 1 8` ; do
    for h in 'lh' 'rh' ; do
        for em in 'fmri-nsd_fsaverage-alexnet' 'fmri-nsd_fsaverage-alexnet_untrained' ; do
            subject_all[$index]=$s
            hemisphere_all[$index]=$h
            encoding_model_all[$index]=$em
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
encoding_model=${encoding_model_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo hemisphere: $hemisphere
echo encoding_model: $encoding_model

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/02-neural_signatures_insilico_validation/vision/fmri/behavioral_modeling

# Run the job
python 02_rsa.py --subject $subject --hemisphere $hemisphere --encoding_model $encoding_model