"""Test the EEG-fMRI encoding fusion models by correlating the t-fMRI responses
for the 200 THINGS EEG2 test images with the corresponding in silico fMRI
responses (independently for each vertex and time point).

Parameters
----------
fmri_subject : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 8.
hemisphere : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import h5py
from tqdm import tqdm
from scipy.stats import pearsonr
from berg import BERG
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Test encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Empty result arrays
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Empty metadata list
metadata = []

n_sub = len(args.fmri_subjects)
n_hemi = len(args.hemispheres)
n_vertex = 163842
n_time = 140

# Empty correlation array of shape:
# (8 subjects, 2 hemispheres, 163842 fMRI vertices, 140 EEG time points)
corr_tfmri_fmri = np.zeros((n_sub, n_hemi, n_vertex, n_time), dtype=np.float32)

# Empty correlation dictionaries
corr_fmri_ncsnr = {}
corr_insilico_fmri_encoding_acc = {}


# =============================================================================
# Loop across subjects and hemispheres
# =============================================================================
for s, sub in enumerate(tqdm(args.fmri_subjects)):

    # Load the subject's metadata
    metadata.append(berg.get_model_metadata(
        'fmri-nsd_fsaverage-huze',
        subject=sub
        ))

    for h, hemi in enumerate(args.hemispheres):


# =============================================================================
# Load the in silico fMRI test responses
# =============================================================================
        data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'insilico_fmri_responses')
        file_name = f'things_eeg_2_test_sub-{sub:02d}_{hemi}'

        fmri_test = h5py.File(os.path.join(data_dir, file_name), 'r')['fmri'][:]


# =============================================================================
# Load the in t-fMRI test responses
# =============================================================================
        data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'tfmri_responses', 'things_eeg_2_test_images')
        file_name = f'tfmri_sub-{sub:02d}_hemi-{hemi}'

        tfmri_test = h5py.File(os.path.join(data_dir, file_name), 'r')['tfmri'][:]


# =============================================================================
# Correlate the t-fMRI with the in silico fMRI responses
# =============================================================================
        # Loop across fMRI vertices
        for v in range(tfmri_test.shape[1]):

            # Center the data
            fmri = fmri_test[:,v] - fmri_test[:,v].mean()
            tfmri = tfmri_test[:,v] - tfmri_test[:,v].mean(axis=0)

            # Normalize the data
            fmri /= np.linalg.norm(fmri)
            tfmri /= np.linalg.norm(tfmri, axis=0)

            # Compute the correlations
            corr_tfmri_fmri[s,h,v] = fmri @ tfmri
            del fmri, tfmri
        del fmri_test, tfmri_test


# =============================================================================
# Correlate the t-fMRI encoding accuracies with the NSD NCSNR and the in silico
# fMRI encoding accuracy
# =============================================================================
        # Three types of correlations are performed:
        # (1) Using all vertices.
        # (2) Using vertices with NCSNR above threshold.
        # (3) Using vertices with NCSNR below threshold.

        # Get the vertex indices for the three correlation types
        threshold = 0.2
        if h == 0:
            ncsnr = metadata[s]['fmri'][f'{hemi}_ncsnr']
            enc_acc = metadata[s]['encoding_models']\
                [f'{hemi}_correlation_nsdcore']
        else:
            ncsnr = np.append(ncsnr, metadata[s]['fmri'][f'{hemi}_ncsnr'])
            enc_acc = np.append(enc_acc, metadata[s]['encoding_models']\
                [f'{hemi}_correlation_nsdcore'])
            correlation = np.append(corr_tfmri_fmri[s,0], corr_tfmri_fmri[s,1],
                1)
            vertex_idx = {}
            vertex_idx['all'] = np.arange(n_vertex*2)
            vertex_idx['below_threshold'] = np.where(ncsnr < 0.2)[0]
            vertex_idx['above_threshold'] = np.where(ncsnr >= 0.2)[0]

            # Loop across correlation types
            for key, val in vertex_idx.items():

                # Empty correlation arrays of shape:
                # (8 subjects, 140 EEG time points)
                if s == 0:
                    corr_fmri_ncsnr[key] = np.zeros((n_sub, n_time),
                        dtype=np.float32)
                    corr_insilico_fmri_encoding_acc[key] = np.zeros((
                        n_sub, n_time), dtype=np.float32)

                # Center the data
                nc = ncsnr[val] - ncsnr[val].mean()
                acc = enc_acc[val] - enc_acc[val].mean()
                corr = correlation[s,val] - correlation[s,val].mean(axis=0)

                # Normalize the data
                nc /= np.linalg.norm(nc)
                acc /= np.linalg.norm(acc)
                corr /= np.linalg.norm(corr, axis=0)

                # Compute the correlations
                corr_fmri_ncsnr[key][s] = nc @ corr
                corr_insilico_fmri_encoding_acc[key][s] = acc @ corr
                del nc, acc, corr
            del ncsnr, enc_acc, correlation


# =============================================================================
# Compute the significance (in silico fMRI vs t-fMRI correlation scores)
# =============================================================================
# Calculate the p-values with t-tests
pval = ttest_1samp(corr_tfmri_fmri, 0, axis=0,
    alternative='two-sided')[1]

# Correct for multiple comparisons
shape = pval.shape
sig_corr_tfmri_fmri = multipletests(pval.flatten(), 0.05, 'fdr_bh')[0]
sig_corr_tfmri_fmri = np.reshape(sig_corr_tfmri_fmri, (shape))


# =============================================================================
# Compute the significance (t-fMRI encoding accuracies vs. NSD NCSNR and in
# silico fMRI encodimg accuracies)
# =============================================================================
# Empty result dictionaries
sig_corr_fmri_ncsnr = {}
sig_corr_insilico_fmri_encoding_acc = {}

# Calculate the p-values with t-tests
for key in corr_fmri_ncsnr.keys():
    pval_corr_fmri_ncsnr = ttest_1samp(corr_fmri_ncsnr[key], 0, axis=0,
        alternative='two-sided')[1]
    pval_corr_insilico_fmri_encoding_acc = ttest_1samp(corr_fmri_ncsnr[key], 0,
        axis=0, alternative='two-sided')[1]

    # Correct for multiple comparisons
    sig_corr_fmri_ncsnr[key] = multipletests(pval_corr_fmri_ncsnr,
        0.05, 'fdr_bh')[0]
    sig_corr_insilico_fmri_encoding_acc[key] = multipletests(
        pval_corr_insilico_fmri_encoding_acc, 0.05, 'fdr_bh')[0]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'metadata': metadata,
    'corr_tfmri_fmri': corr_tfmri_fmri,
    'corr_fmri_ncsnr': corr_fmri_ncsnr,
    'corr_insilico_fmri_encoding_acc': corr_insilico_fmri_encoding_acc,
    'sig_corr_tfmri_fmri': sig_corr_tfmri_fmri,
    'sig_corr_fmri_ncsnr': sig_corr_fmri_ncsnr,
    'sig_corr_insilico_fmri_encoding_acc': sig_corr_insilico_fmri_encoding_acc
}

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'encoding_fusion_accuracy')
os.makedirs(save_dir, exist_ok=True)

file_name = 'encoding_fusion_accuracy'

np.save(os.path.join(save_dir, file_name), results)


























































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
