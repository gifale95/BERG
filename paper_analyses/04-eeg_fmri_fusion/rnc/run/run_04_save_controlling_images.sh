#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-04_eeg_fmri_fusion-rnc-04_save_controlling_images
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=01:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a roi_all
index=0
for r in 'V1' 'hV4' 'ventral' ; do
    roi_all[$index]=$r
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
echo roi: $roi

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/rnc

# Run the job
python 04_save_controlling_images.py --roi $roi