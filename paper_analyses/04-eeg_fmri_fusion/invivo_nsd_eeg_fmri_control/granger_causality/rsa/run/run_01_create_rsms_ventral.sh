#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-invivo_nsd_eeg_fmri_control-granger_causality-RSA-01_create_rsms_ventral
#SBATCH --mail-type=end
#SBATCH --mem=100000
#SBATCH --time=30:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a roi_all
index=0
for fs in '1' '4' '5' '6' '7' '8' ; do
    for r in 'ventral' ; do
        subject_all[$index]=$fs
        roi_all[$index]=$r
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo roi: $roi

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/invivo_nsd_eeg_fmri_control/granger_causality/rsa

# Run the job
python 01_create_rsms.py --subject $subject --roi $roi