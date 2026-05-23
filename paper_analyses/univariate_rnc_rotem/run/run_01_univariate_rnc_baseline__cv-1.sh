#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=univariate_rnc_rotem-01_univariate_rnc_baseline__cv-1
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=02:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a cv_subject_all
declare -a roi_all
index=0
for cs in `seq 1 8` ; do
    for r in 'V1' 'ventral' ; do
        cv_subject_all[$index]=$cs
        roi_all[$index]=$r
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
cv_subject=${cv_subject_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
echo cv_subject: $cv_subject
echo roi: $roi

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/univariate_rnc_rotem

# Run the job
python 01_univariate_rnc_baseline.py --cv '1' --cv_subject $cv_subject --roi $roi