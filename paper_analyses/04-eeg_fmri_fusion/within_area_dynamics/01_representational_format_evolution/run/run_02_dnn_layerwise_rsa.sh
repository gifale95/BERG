#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-04_eeg_fmri_fusion-within_area_dynamics-01_representational_format_evolution-02_dnn_layerwise_rsa
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=01:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a fmri_subject_all
declare -a roi_all
declare -a images_all
declare -a dnn_all
index=0
for fs in `seq 1 8` ; do
    for r in 'V1' 'V2' 'V3' 'hV4' 'FFA' 'EBA' 'PPA' ; do
        for i in 'things_eeg_2_vivo' 'things_eeg_2_silico' 'nsd_515_shared' ; do
            for d in 'dinov2l' ; do
                fmri_subject_all[$index]=$fs
                roi_all[$index]=$r
                images_all[$index]=$i
                dnn_all[$index]=$d
                ((index=index+1))
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
images=${images_all[$SLURM_ARRAY_TASK_ID]}
dnn=${dnn_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo roi: $roi
echo images: $images
echo dnn: $dnn

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/within_area_dynamics/01_representational_format_evolution

# Run the job
python 02_dnn_layerwise_rsa.py --fmri_subject $fmri_subject --roi $roi --images $images --dnn $dnn