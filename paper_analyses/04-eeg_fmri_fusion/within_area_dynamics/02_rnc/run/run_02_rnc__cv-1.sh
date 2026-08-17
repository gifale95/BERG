#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-04_eeg_fmri_fusion-within_area_dynamics-02_rnc-02_rnc__cv-1
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=00:10:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a cv_subject_all
declare -a roi_all
declare -a time_window_pair_all
declare -a imageset_all
index=0
for cs in `seq 1 8` ; do
    for r in 'V1' 'V2' 'V3' 'hV4' 'FFA' 'EBA' 'PPA' ; do
        for t in '0.06-0.10__0.20-0.25' ; do
            for i in 'imagenet' 'coco' ; do
                cv_subject_all[$index]=$cs
                roi_all[$index]=$r
                time_window_pair_all[$index]=$t
                imageset_all[$index]=$i
                ((index=index+1))
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
cv_subject=${cv_subject_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
time_window_pair=${time_window_pair_all[$SLURM_ARRAY_TASK_ID]}
imageset=${imageset_all[$SLURM_ARRAY_TASK_ID]}
echo cv_subject: $cv_subject
echo roi: $roi
echo time_window_pair: $time_window_pair
echo imageset: $imageset

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/within_area_dynamics/02_rnc

# Run the job
python 02_rnc.py --cv '1' --cv_subject $cv_subject --roi $roi --time_window_pair $time_window_pair --imageset $imageset