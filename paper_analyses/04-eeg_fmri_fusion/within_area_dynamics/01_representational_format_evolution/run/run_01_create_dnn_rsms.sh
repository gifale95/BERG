#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-04_eeg_fmri_fusion-within_area_dynamics-01_representational_format_evolution-01_create_dnn_rsms
#SBATCH --mail-type=end
#SBATCH --mem=50000
#SBATCH --time=01:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a dnn_all
declare -a images_all
index=0
for d in 'dinov2l' ; do
    for i in 'things_eeg_2' 'nsd_515_shared' ; do
        dnn_all[$index]=$d
        images_all[$index]=$i
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
dnn=${dnn_all[$SLURM_ARRAY_TASK_ID]}
images=${images_all[$SLURM_ARRAY_TASK_ID]}
echo dnn: $dnn
echo images: $images

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/within_area_dynamics/01_representational_format_evolution

# Run the job
python 01_create_dnn_rsms.py --dnn $dnn --images $images