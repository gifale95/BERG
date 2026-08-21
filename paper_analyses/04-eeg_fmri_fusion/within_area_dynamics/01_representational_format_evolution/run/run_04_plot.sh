#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-04_eeg_fmri_fusion-within_area_dynamics-01_representational_format_evolution-04_plot
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=00:30:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a images_all
declare -a dnn_all
index=0
for i in 'things_eeg_2_vivo' ; do
    for d in 'dinov2l' ; do
        images_all[$index]=$i
        dnn_all[$index]=$d
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
images=${images_all[$SLURM_ARRAY_TASK_ID]}
dnn=${dnn_all[$SLURM_ARRAY_TASK_ID]}
echo images: $images
echo dnn: $dnn

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/within_area_dynamics/01_representational_format_evolution

# Run the job
python 04_plot.py --images $images --dnn $dnn