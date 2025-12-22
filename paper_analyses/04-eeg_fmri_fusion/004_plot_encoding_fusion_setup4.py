"""Plot the encoding fusion results on flattened cortical surfaces.

This script loads correlation results from the encoding fusion testing and creates
visualizations on cortical surfaces using pycortex. It creates plots for both
individual fMRI subjects and averages across subjects.

Parameters
----------
fmri_subjects : list of int
    List of the used NSD fMRI subjects (1-8).
model_name : str
    Name of the model configuration.
experiment_name : str
    Name for experiment-specific results folder (must match training/testing).
nest_dir : str
    Neural encoding simulation toolkit directory.

"""

import argparse
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import cortex
import cortex.polyutils


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', nargs='+', type=int, default=[1, 2],
                   help='List of fMRI subjects to plot')
parser.add_argument('--model_name', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--experiment_name', type=str, default='default_experiment')
parser.add_argument('--nest_dir', 
                   default='/pfss/mlde/workspaces/mlde_wsp_PI_Roig/bersch/repositories/BERG/brain-encoding-response-generator', 
                   type=str)
args = parser.parse_args()

print('>>> Plot encoding fusion results <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the encoding fusion correlation results
# =============================================================================
n_vertices_all = 327684
n_vertices_hemi = 163842
n_time = 140

# Arrays of shape: (fMRI subjects × Time × Vertices)
correlations = np.zeros((len(args.fmri_subjects), n_time, n_vertices_all), 
                        dtype=np.float32)

# Directory where test results are stored
results_dir = os.path.join(args.nest_dir, 'results', 'paper_analyses',
    'encoding_fusion', 'test_results', args.experiment_name, 
    str(args.model_name), 'aggr-append', 'regression-linear')

print(f"Loading results from: {results_dir}")

# Load results for each fMRI subject
for fs, fmri_sub in enumerate(args.fmri_subjects):
    # Load left hemisphere results
    lh_file_name = f'fmri_sub-{fmri_sub:02d}_lh.npy'
    lh_results = np.load(os.path.join(results_dir, lh_file_name),
                        allow_pickle=True).item()
    
    # Load right hemisphere results
    rh_file_name = f'fmri_sub-{fmri_sub:02d}_rh.npy'
    rh_results = np.load(os.path.join(results_dir, rh_file_name),
                        allow_pickle=True).item()
    
    # Combine left and right hemisphere correlations
    correlations[fs, :, :n_vertices_hemi] = lh_results['correlations']
    correlations[fs, :, n_vertices_hemi:] = rh_results['correlations']
    
    # Get times (same for all subjects)
    times = lh_results['times']

print(f"Loaded correlations shape: {correlations.shape}")
print(f"Time points: {len(times)}")


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 40
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 1
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'

# Pycortex subject
subject = 'fsaverage'


# =============================================================================
# Plot the results for individual fMRI subjects
# =============================================================================
# Save directory
save_dir = os.path.join(args.nest_dir, 'results', 'paper_analyses',
    'encoding_fusion', 'results_plots', args.experiment_name, 
    str(args.model_name), 'aggr-append', 
    'regression-linear', 'fmri-ind')

os.makedirs(save_dir, exist_ok=True)

print(f"Plotting individual subject results to: {save_dir}")

# Loop over time and fMRI subjects
for t, time in enumerate(tqdm(times, desc="Plotting individual subjects")):
    for fs, fmri_sub in enumerate(args.fmri_subjects):
        # Get correlation data for this subject and timepoint
        data = correlations[fs, t, :]
        
        # Create vertex data for pycortex
        vertex_data = cortex.Vertex(data, subject, cmap='RdBu_r', vmin=-1, vmax=1,
                                    with_colorbar=True)
        
        # Create cortical surface plot
        fig = cortex.quickshow(vertex_data,
                              with_curvature=True,
                              curvature_brightness=0.5,
                              with_rois=True,
                              with_labels=False,
                              linewidth=3,
                              linecolor=(1, 1, 1),
                              with_colorbar=True)
        
        # Add title and save
        title = f'Time (s): {np.round(time, 2)}'
        plt.title(title, fontsize=fontsize)
        
        plot_file = os.path.join(save_dir, 
                                f'correlation_fmri_sub-{fmri_sub:02d}_time-{t:03d}.jpg')
        plt.savefig(plot_file, dpi=100, bbox_inches='tight', format='jpg')
        plt.close()


# =============================================================================
# Plot the results averaged across fMRI subjects
# =============================================================================
# Save directory
save_dir = os.path.join(args.nest_dir, 'results', 'paper_analyses',
    'encoding_fusion', 'results_plots', args.experiment_name, 
    str(args.model_name), 'aggr-append', 
    'regression-linear', 'fmri-avg')

os.makedirs(save_dir, exist_ok=True)

print(f"Plotting averaged results to: {save_dir}")

# Loop over time
for t, time in enumerate(tqdm(times, desc="Plotting averaged across subjects")):
    # Average correlations across all fMRI subjects
    data = np.mean(correlations[:, t, :], axis=0)
    
    # Create vertex data for pycortex
    vertex_data = cortex.Vertex(data, subject, cmap='RdBu_r', vmin=-1, vmax=1,
                                with_colorbar=True)
    
    # Create cortical surface plot
    fig = cortex.quickshow(vertex_data,
                          with_curvature=True,
                          curvature_brightness=0.5,
                          with_rois=True,
                          with_labels=False,
                          linewidth=3,
                          linecolor=(1, 1, 1),
                          with_colorbar=True)
    
    # Add title and save
    title = f'Time (s): {np.round(time, 2)}'
    plt.title(title, fontsize=fontsize)
    
    plot_file = os.path.join(save_dir, f'correlation_avg_time-{t:03d}.jpg')
    plt.savefig(plot_file, dpi=100, bbox_inches='tight', format='jpg')
    plt.close()
