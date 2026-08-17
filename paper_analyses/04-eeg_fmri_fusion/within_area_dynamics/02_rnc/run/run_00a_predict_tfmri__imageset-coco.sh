#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-04_eeg_fmri_fusion-within_area_dynamics-02_rnc-00a_predict_tfmri__coco
#SBATCH --mail-type=end
#SBATCH --mem=30000
#SBATCH --time=05:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a fmri_subject_all
declare -a roi_all
declare -a imageset_all
declare -a current_batch_all
index=0
for fs in `seq 1 8` ; do
    for r in 'V1' 'V2' 'V3' 'hV4' 'FFA' 'EBA' 'PPA' ; do
        for i in 'coco' ; do
            for b in `seq 0 9` ; do
                fmri_subject_all[$index]=$fs
                roi_all[$index]=$r
                imageset_all[$index]=$i
                current_batch_all[$index]=$b
                ((index=index+1))
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
imageset=${imageset_all[$SLURM_ARRAY_TASK_ID]}
current_batch=${current_batch_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo roi: $roi
echo imageset: $imageset
echo current_batch: $current_batch

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/within_area_dynamics/02_rnc

# Run the job
python 00a_predict_tfmri.py --fmri_subject $fmri_subject --roi $roi --imageset $imageset --current_batch $current_batch