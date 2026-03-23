#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-neural_control-01_generate_insilico_responses
#SBATCH --mail-type=end
#SBATCH --mem=25000
#SBATCH --time=03-00:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a roi_all
index=0
for s in 'N' 'F' ; do
    for r in 'V1' 'V4' 'IT' ; do
        subject_all[$index]=$s
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
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/05_neural_control

# Run the job
python 01_generate_insilico_responses.py --subject $subject --roi $roi