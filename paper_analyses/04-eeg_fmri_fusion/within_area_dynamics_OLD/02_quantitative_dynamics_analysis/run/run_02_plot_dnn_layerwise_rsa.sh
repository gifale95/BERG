#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=berg-04_eeg_fmri_fusion-neural_dynamics-02_quantitative_dynamics_analysis-02_plot_dnn_layerwise_rsa
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=00:30:00
#SBATCH --qos=extended

# Create the parameters combinations
declare -a roi_all
declare -a time_window_pair_all
declare -a use_time_bins_all
declare -a correlation_measure_all
declare -a dnn_all
declare -a normalize_all
index=0
for r in 'V1' 'hV4' 'FFA' ; do
    for tw in '0.05-0.10__0.20-0.25' ; do
        for tb in '0' '1' ; do
            for cm in 'pearson' ; do
                for d in 'dinov2l' ; do
                    for n in '0' '1' ; do
                        roi_all[$index]=$r
                        time_window_pair_all[$index]=$tw
                        use_time_bins_all[$index]=$tb
                        correlation_measure_all[$index]=$cm
                        dnn_all[$index]=$d
                        normalize_all[$index]=$n
                        ((index=index+1))
                    done
                done
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
time_window_pair=${time_window_pair_all[$SLURM_ARRAY_TASK_ID]}
use_time_bins=${use_time_bins_all[$SLURM_ARRAY_TASK_ID]}
correlation_measure=${correlation_measure_all[$SLURM_ARRAY_TASK_ID]}
dnn=${dnn_all[$SLURM_ARRAY_TASK_ID]}
normalize=${normalize_all[$SLURM_ARRAY_TASK_ID]}
echo roi: $roi
echo time_window_pair: $time_window_pair
echo use_time_bins: $use_time_bins
echo correlation_measure: $correlation_measure
echo dnn: $dnn
echo normalize: $normalize

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Change to the .py script directory
cd /home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/neural_dynamics/02_quantitative_dynamics_analysis

# Run the job
python 02_plot_dnn_layerwise_rsa.py --roi $roi --time_window_pair $time_window_pair --use_time_bins $use_time_bins --correlation_measure $correlation_measure --dnn $dnn --normalize $normalize