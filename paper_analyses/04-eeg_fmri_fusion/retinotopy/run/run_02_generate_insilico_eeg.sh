#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-retinotopy-02_generate_insilico_eeg
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=00:45:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a test_img_all
index=0
for i in `seq 0 99` ; do
    test_img_all[$index]=$i
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
test_img=${test_img_all[$SLURM_ARRAY_TASK_ID]}
echo test_img: $test_img

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/retinotopy

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 02_generate_insilico_eeg.py --test_img $test_img