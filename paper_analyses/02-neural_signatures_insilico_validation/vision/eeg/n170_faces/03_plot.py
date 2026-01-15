"""Plot the ERPs for faces and objects.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subjects : list
    List of EEG subject identifiers.
channels : list
    List containing the EEG channel type(s) retained for the analyses.
    Possible values are: 'O' (occipital), 'P' (posterior), 'T' (temporal),
    'C' (central), 'F' (frontal). Alternatively, the list can also contain the
    names of the individual channels used.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
parser.add_argument('--channels', default=['P7', 'P8', 'PO7', 'PO8', 'TP7', 'TP8'], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'n170_faces', 'plots', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the pairwise decoding results
# =============================================================================
results_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',  'n170_faces',
    'stats', args.encoding_model, 'stats_channels-'+'-'.join(args.channels)+
    '.npy')

results = np.load(results_dir, allow_pickle=True).item()

erp_faces = results['erp_faces']
erp_objects = results['erp_objects']
ci_erp_faces = results['ci_erp_faces']
ci_erp_objects = results['ci_erp_objects']
pval_erp_diff = results['pval_erp_diff']
pval_erp_diff_corrected = results['pval_erp_diff_corrected']
sig_erp_diff = results['sig_erp_diff']
times = results['metadata'][0]['eeg']['times']


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 25
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams["font.weight"] = "normal"
matplotlib.rcParams["axes.labelweight"] = "normal"
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 0
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 0
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
colors = [(139/255, 0/255, 0/255), (0/255, 0/255, 0/255)]


# =============================================================================
# Plot the ERPs
# =============================================================================
fig = plt.figure(figsize=(10, 7.5))

# Plot the stimulus onset dashed line
plt.plot([0, 0], [100, -100], 'k--', linewidth=2, alpha=.5, label='_nolegend_')

# Plot the ERPs
plt.plot(times, np.mean(erp_faces, 0), color=colors[0], linewidth=2,
    label='Faces')
plt.plot(times, np.mean(erp_objects, 0), color=colors[1], linewidth=2,
    label='Objects')

# Plot the CIs
plt.fill_between(times, ci_erp_faces[1], ci_erp_faces[0], color=colors[0],
    alpha=.1)
plt.fill_between(times, ci_erp_objects[1], ci_erp_objects[0], color=colors[1],
    alpha=.1)

# Plot the significance markers
sig = np.empty(len(times))
sig[:] = np.nan
sig[sig_erp_diff] = -1.25
plt.scatter(times, sig, s=100, color=colors[0])

# x-axis parameters
plt.xlabel('Time (ms)', fontsize=fontsize)
xticks = [-0.1, 0, .1, .2, .3, .4, .5, .595]
xlabels = [-100, 0, 100, 200, 300, 400, 500, 600]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
plt.ylabel('μV', fontsize=fontsize)
yticks = [-1, -.5, 0, .5]
ylabels = [-1, -.5, 0, .5]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-1.4, top=.5)

# Legend
plt.legend(ncol=1, fontsize=fontsize, loc=4, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'erps_channels-'+'-'.join(args.channels)+
    '.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close(fig)