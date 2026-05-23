#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=univariate_rnc_rotem-00c_predict_fmri__roi-ventral
#SBATCH --mail-type=end
#SBATCH --mem=50000
#SBATCH --time=15:00:00
#SBATCH --qos=extended
#SBATCH --partition=agcichy
#SBATCH --gres=gpu:1 # number of GPUs

# CUDA module
module add CUDA/12.4.0

# Create the parameters combinations
declare -a fmri_subject_all
declare -a roi_all
index=0
for fs in `seq 1 8` ; do
    for r in 'ventral' ; do
        fmri_subject_all[$index]=$fs
        roi_all[$index]=$r
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo roi: $roi

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/univariate_rnc_rotem

# Run the job
python 00c_predict_fmri.py --fmri_subject $fmri_subject --roi $roi --berg_dir '/scratch/giffordale95/projects/brain-encoding-response-generator'