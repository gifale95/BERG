#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_fmri_fusion-invivo_nsd_eeg_fmri_control-dnn_llm_modeling-RSA-01_rsa
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=20:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a hemisphere_all
declare -a cv_split_all
declare -a time_split_all
index=0
for fs in '1' '4' '5' '6' '7' '8' ; do
    for h in 'lh' 'rh' ; do
        for cv in '1' '2' ; do
            for t in `seq 0 9` ; do
                subject_all[$index]=$fs
                hemisphere_all[$index]=$h
                cv_split_all[$index]=$cv
                time_split_all[$index]=$t
                ((index=index+1))
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
cv_split=${cv_split_all[$SLURM_ARRAY_TASK_ID]}
time_split=${time_split_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo hemisphere: $hemisphere
echo cv_split: $cv_split
echo time_split: $time_split

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/invivo_nsd_eeg_fmri_control/dnn_llm_modeling/02_rsa

# Run the job
python 01_rsa.py --subject $subject --hemisphere $hemisphere --cv_split $cv_split --time_split $time_split