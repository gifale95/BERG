#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=03-rnc_eeg-02_create_rsms
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=06:00:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a subject_all
declare -a time_all
declare -a rsm_split_all
index=0
for s in `seq 1 10` ; do
    for t in '0.1' '0.2' '0.3' '0.4' ; do
        for r in `seq 1 5` ; do
            subject_all[$index]=$s
            time_all[$index]=$t
            rsm_split_all[$index]=$r
          ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
time=${time_all[$SLURM_ARRAY_TASK_ID]}
rsm_split=${rsm_split_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo time: $time
echo rsm_split: $rsm_split

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/03-rnc_eeg

# Run the job
python 02_create_rsms.py --subject $subject --time $time --rsm_split $rsm_split