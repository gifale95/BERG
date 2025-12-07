"""Plot the ERPs for faces and objects.

Parameters
----------
subjects : list
    List of the subject identifiers for the MEG encoding models. Since the
    used encoding models are trained on THINGS MEG1 data, valid subject
    identifiers are integers from 1 to 4.
sensors : list
    List containing the MEG sensor names retained for the analyses.
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
parser.add_argument('--subjects', default=[1, 2, 3, 4], type=int)
parser.add_argument('--sensors', default=['MLT23', 'MLT24', 'MLT33', 'MLT34', 'MRT23', 'MRT24', 'MRT33', 'MRT34'], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'meg', 'm170_faces', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the pairwise decoding results
# =============================================================================
results_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'meg',  'm170_faces',
    'stats', 'stats_'+'sensors-'+'-'.join(args.sensors)+'.npy')

results = np.load(results_dir, allow_pickle=True).item()

erp_faces = results['erp_faces']
erp_objects = results['erp_objects']
ci_erp_faces = results['ci_erp_faces']
ci_erp_objects = results['ci_erp_objects']
pval_erp_diff = results['pval_erp_diff']
pval_erp_diff_corrected = results['pval_erp_diff_corrected']
sig_erp_diff = results['sig_erp_diff']
times = results['metadata'][0]['meg']['times']


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 30
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 1
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
fig= plt.figure(figsize=(13, 7))

# Plot the stimulus onset dashed line
plt.plot([0, 0], [100, -100], 'k--', linewidth=3, alpha=.5, label='_nolegend_')

# Plot the ERPs
plt.plot(times, np.mean(erp_faces, 0), color=colors[0], linewidth=3,
    label='Faces')
plt.plot(times, np.mean(erp_objects, 0), color=colors[1], linewidth=3,
    label='Objects')

# Plot the CIs
plt.fill_between(times, ci_erp_faces[1], ci_erp_faces[0], color=colors[0],
    alpha=.2)
plt.fill_between(times, ci_erp_objects[1], ci_erp_objects[0], color=colors[1],
    alpha=.2)

# Plot the significance markers
sig = np.empty(len(times))
sig[:] = np.nan
sig[sig_erp_diff] = -1.25
plt.scatter(times, sig, s=100, color=colors[0])

# x-axis parameters
plt.xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, 100, 200, 300, 400, 500]
plt.xticks(ticks=xticks, labels=xlabels) # type: ignore
plt.xlim(left=min(times), right=0.5)

# y-axis parameters
plt.ylabel('fT', fontsize=fontsize)
yticks = [-1.5, -1, -.5, 0, .5]
ylabels = [-1.5, -1, -.5, 0, .5]
plt.yticks(ticks=yticks, labels=ylabels) # type: ignore
plt.ylim(bottom=-.5, top=.5)

# Legend
plt.legend(ncol=1, fontsize=fontsize, loc=4, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'erps_sensors-'+'-'.join(args.sensors)+
    '.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')