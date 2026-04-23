"""Compute Granger Causality, using RSA, on in silico neural responses found
through neural control.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico
    responses.
subjects : list
    The subject identifiers for the monkey encoding model. Since the used
    encoding models are trained on the TVSD data, valid subject identifiers
    are "N" and "F".
roi_target: str
    The target ROI for computing Granger Causality. Valid values are "V1",
    "V4", and "IT".
roi_source: str
    The source ROI for computing Granger Causality. Valid values are "V1",
    "V4", and "IT".
rois_neural_control: str
    If 'single', use images from neural control applied to only the source ROI.
    If 'both', use images from neural control applied to both the source and
    target ROIs.
cv: int
    If 1, cross-validate the controlling images across the two monkyes.
    If 0, do not cross-validate.
time_window_ms : int
    Time window in milliseconds for computing Granger Causality.
offset_ms : int
    Offset in milliseconds for the time window.
regression : str
    The type of regression to use for computing Granger Causality. If 'linear',
    use linear regression. If 'ridge', use RidgeCV regression.
berg_dir : str
    Directory of the BERG.
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subjects', default=['F', 'N'], type=list)
parser.add_argument('--roi_target', default='V1', type=str)
parser.add_argument('--roi_source', default='V4', type=str)
parser.add_argument('--rois_neural_control', default=['single', 'both'], type=list)
parser.add_argument('--objectives', default=['max', 'baseline', 'min'], type=list)
parser.add_argument('--cv', default=[0, 1], type=list)
parser.add_argument('--time_window_ms', default=100, type=int)
parser.add_argument('--offset_ms', default=20, type=int)
parser.add_argument('--regression', default='linear', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/ccn_datasets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Load the results
# =============================================================================
gc = {}

data_dir = os.path.join(args.berg_dir, 'neural_control',
    'granger_causality_neural_control', 'granger_causality',
    args.encoding_model)

for sub in args.subjects:
    for rois_nc in args.rois_neural_control:
        for obj in args.objectives:
            for cv in args.cv:

                file_name = (f'gc_sub-{sub}_roi_target-{args.roi_target}_'
                    f'roi_source-{args.roi_source}_rois_neural_control-'
                    f'{rois_nc}_objective-{obj}_cv-{cv}_'
                    f'time_window_ms-{args.time_window_ms:03d}_offset_ms-'
                    f'{args.offset_ms:03d}_regression-{args.regression}.npy')

                data = np.load(os.path.join(data_dir, file_name),
                    allow_pickle=True).item()

                gc[(sub, rois_nc, obj, cv)] = data['gc']
                times = data['times']
                idx_t_start = data['idx_t_start']


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
colors = [(139/255, 0/255, 0/255), (0/255, 0/255, 0/255),
    (0/255, 115/255, 155/255)]


# =============================================================================
# Plot the results
# =============================================================================
times_gc = times[idx_t_start:]

for s, sub in enumerate(args.subjects):

    fig, axs = plt.subplots(len(args.cv), len(args.rois_neural_control),
        sharex=True, sharey=True, figsize=(20, 15)) # (10, 7.5)

    for c, cv in enumerate(args.cv):
        for r, rois_nc in enumerate(args.rois_neural_control):

            # Plot the chance onset dashed line
            axs[c,r].plot([-1000, 1000], [0, 0], 'k--', linewidth=2,
                alpha=.25, label='_nolegend_')

            # PLot the GC scores
            for o, obj in enumerate(args.objectives):
                gc_score = gc[(sub, rois_nc, obj, cv)]
                axs[c,r].plot(times_gc, gc_score, color=colors[o],
                    linewidth=2, label=obj)

            # Title
            title = f'Sub-{sub}, CV-{cv}, rois_nc-{rois_nc}'
            axs[c,r].set_title(title, fontsize=fontsize)

            # x-axis parameters
            if c == len(args.cv)-1:
                axs[c,r].set_xlabel('Time (ms)', fontsize=fontsize)
                # xticks = [-100, -50, 0, 50, 100, 150, 199]
                # xlabels = [-100, -50, 0, 50, 100, 150, 200]
                # axs[c,r].set_xticks(ticks=xticks, labels=xlabels)
                axs[c,r].set_xlim(left=min(times_gc), right=max(times_gc))

            # y-axis parameters
            if r == 0:
                axs[c,r].set_ylabel('Granger Causality', fontsize=fontsize)
                # yticks = [10, 15, 20, 25, 30]
                # ylabels = [10, 15, 20, 25, 30]
                # axs[c,r].set_yticks(ticks=yticks, labels=ylabels)
                # axs[c,r].set_ylim(bottom=8, top=29)

            # Legend
            if c == 0 and r == 0:
                axs[c,r].legend(ncol=1, fontsize=fontsize, loc=0,
                    frameon=False)

    # Save the figure
    save_dir = os.path.join(args.berg_dir, 'neural_control',
        'granger_causality_neural_control', 'plots')
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir,
        f'granger_causality_neural_control_sub-{sub}.svg')
    fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
    plt.close(fig)