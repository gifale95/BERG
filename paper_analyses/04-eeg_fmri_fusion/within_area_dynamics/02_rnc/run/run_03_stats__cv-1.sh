#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-04_eeg_fmri_fusion-within_area_dynamics-02_rnc-03_stats__cv-1
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:30:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a roi_all
declare -a time_window_pair_all
declare -a imageset_all
index=0
# for r in 'V1' 'V2' 'V3' 'hV4' 'FFA' 'OFA' 'EBA' 'FBA' 'PPA' 'OPA' ; do
for r in 'V1' 'FFA' 'OFA' 'EBA' 'FBA' 'PPA' 'OPA' ; do
    # for t in '0.06-0.10__0.20-0.25' '0.06-0.10__0.25-0.30' '0.06-0.10__0.30-0.35' '0.06-0.10__0.35-0.40' '0.10-0.15__0.20-0.25' '0.10-0.15__0.25-0.30' '0.10-0.15__0.30-0.35' '0.10-0.15__0.35-0.40' '0.06-0.15__0.15-0.25' ; do
    for t in '0.06-0.10__0.20-0.25' ; do
        # for i in 'imagenet' 'coco' ; do
        for i in 'coco' ; do
            roi_all[$index]=$r
            time_window_pair_all[$index]=$t
            imageset_all[$index]=$i
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
time_window_pair=${time_window_pair_all[$SLURM_ARRAY_TASK_ID]}
imageset=${imageset_all[$SLURM_ARRAY_TASK_ID]}
echo roi: $roi
echo time_window_pair: $time_window_pair
echo imageset: $imageset

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/within_area_dynamics/02_rnc

# Run the job
python 03_stats.py --cv '1' --roi $roi --time_window_pair $time_window_pair --imageset $imageset