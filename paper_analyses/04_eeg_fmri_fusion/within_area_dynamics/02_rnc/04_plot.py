"""Plot the univariate RNC cross-subject validated results.

Parameters
----------
roi: str
   Used ROI.
time_window_pair: str
   A string specifying the two time windows of interest.
imageset : str
    The image set to use for the analysis. Possible values are:
    'imagenet_train' (ILSVRC-2012 training split),
    'imagenet_val' (ILSVRC-2012 validation split),
    'imagenet_all' (ILSVRC-2012 training + validation split),
    'coco' (MS COCO 2017 test split).
berg_dir : str
   Directory of the BERG.

"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

parser = argparse.ArgumentParser()
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--time_window_pair', default='0.06-0.10__0.20-0.25', type=str)
parser.add_argument('--imageset', default='imagenet_all', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Break down the time windows
# =============================================================================
time_window_1_start, time_window_1_end = map(
   float, args.time_window_pair.split('__')[0].split('-'))
time_window_1_start = int(time_window_1_start * 1000)
time_window_1_end = int(time_window_1_end * 1000)
time_window_2_start, time_window_2_end = map(
   float, args.time_window_pair.split('__')[1].split('-'))
time_window_2_start = int(time_window_2_start * 1000)
time_window_2_end = int(time_window_2_end * 1000)

tw_1 = f'{time_window_1_start}-{time_window_1_end} ms'
tw_2 = f'{time_window_2_start}-{time_window_2_end} ms'
tw_all = '60-400 ms'


# =============================================================================
# Load the neural control results
# =============================================================================
# Load the results
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
   'within_area_dynamics', 'rnc', 'stats', 'cv-1', args.time_window_pair,
   f'imageset-{args.imageset}', f'stats_roi-{args.roi}.npy')
stats = np.load(data_dir, allow_pickle=True).item()

# Get the univariate responses for baseline and controlling images
resp_high_all = stats['cv_resp_all']['nc_high_all']
resp_tw_1_h1l2 = stats['cv_resp_1']['rnc_high_1_low_2']
resp_tw_1_l1h2 = stats['cv_resp_1']['rnc_low_1_high_2']
resp_tw_2_h1l2 = stats['cv_resp_2']['rnc_high_1_low_2']
resp_tw_2_l1h2 = stats['cv_resp_2']['rnc_low_1_high_2']
base_resp_1 = stats['base_resp_1']
base_resp_2 = stats['base_resp_2']
base_resp_all = stats['base_resp_all']
nc_between_subj_pval = stats['nc_between_subj_pval']
rnc_between_subj_pval_h1l2_tw_1 = stats['rnc_between_subj_pval_h1l2_tw_1']
rnc_between_subj_pval_l1h2_tw_1 = stats['rnc_between_subj_pval_l1h2_tw_1']
rnc_between_subj_pval_h1l2_tw_2 = stats['rnc_between_subj_pval_h1l2_tw_2']
rnc_between_subj_pval_l1h2_tw_2 = stats['rnc_between_subj_pval_l1h2_tw_2']

# # Load the image statistics
# data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
#    'within_area_dynamics', 'rnc', 'stats', 'cv-0', args.time_window_pair,
#    f'imageset-{args.imageset}', f'stats_images_roi-{args.roi}.npy')
# stats_img = np.load(data_dir, allow_pickle=True).item()
# img_complexity_png = stats_img['img_complexity_png']
# img_complexity_jpg = stats_img['img_complexity_jpg']
# img_complexity_std = stats_img['img_complexity_std']
# img_complexity_best_layer = stats_img['img_complexity_best_layer']
# p_val_std = stats_img['p_val_std']
# p_val_best_layer = stats_img['p_val_best_layer']
# p_val_png = stats_img['p_val_png']
# p_val_jpg = stats_img['p_val_jpg']

# # Load the image statistics (NEW ICNET)
# data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
#    'within_area_dynamics', 'rnc', 'stats', 'cv-0', args.time_window_pair,
#    f'imageset-{args.imageset}', f'stats_images_roi-{args.roi}_method-icnet.npy')
# stats_img = np.load(data_dir, allow_pickle=True).item()
# img_complexity_icnet = stats_img['img_complexity']
# p_val_icnet = stats_img['p_val']

# # Load the image statistics (NEW nagle_lavie)
# data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
#    'within_area_dynamics', 'rnc', 'stats', 'cv-0', args.time_window_pair,
#    f'imageset-{args.imageset}', f'stats_images_roi-{args.roi}_method-nagle_lavie.npy')
# stats_img = np.load(data_dir, allow_pickle=True).item()
# img_complexity_nagle_lavie = stats_img['img_complexity']
# p_val_nagle_lavie = stats_img['p_val']


# =============================================================================
# Create the plot save directory
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
   'within_area_dynamics', 'rnc', 'plots', f'imageset-{args.imageset}')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Plot helper functions
# =============================================================================
def marginal_kde(samples, grid, bw_method='scott'):
   """Gaussian KDE of 1D samples evaluated on grid.

   Parameters
   ----------
   samples : np.ndarray
      1D array of samples, non-finite values are removed.
   grid : np.ndarray
      1D array of evaluation points.
   bw_method : str or float
      Bandwidth selector passed to scipy.stats.gaussian_kde.

   Returns
   -------
   density : np.ndarray
      Estimated density at grid, zeros if the KDE is not defined.
   """
   samples = samples[np.isfinite(samples)] # gaussian_kde silently breaks on NaN
   if len(samples) < 2 or np.std(samples) == 0:
      return np.zeros(len(grid), dtype=np.float32)
   kde = gaussian_kde(samples, bw_method=bw_method)
   return kde(grid).astype(np.float32)

def p_to_asterisks(p_value, alpha_levels=(.01, .001, .0001)):
   """Convert a p-value to an asterisk string.

   Parameters
   ----------
   p_value : float
      Two-tailed p-value.
   alpha_levels : tuple
      Thresholds, one asterisk per threshold crossed.

   Returns
   -------
   asterisks : str
      '*', '**', '***', or 'n.s.'.
   """
   if not np.isfinite(p_value):
      return 'n.s.'
   n_stars = int(np.sum(p_value < np.asarray(alpha_levels)))
   return '*' * n_stars if n_stars > 0 else 'n.s.'


def anchor_positions(density, grid, samples, anchor):
   """Bracket anchor position and the density height at that position.

   Parameters
   ----------
   density : np.ndarray
      Densities, shape (n_conditions, n_grid).
   grid : np.ndarray
      Evaluation points, shape (n_grid).
   samples : list
      Per-condition 1D sample arrays, used only if anchor is 'mean'.
   anchor : str
      'mode' or 'mean'.

   Returns
   -------
   pos : np.ndarray
      Anchor position on the response axis, shape (n_conditions).
   tip : np.ndarray
      Density at pos, i.e. where the bracket leg meets the curve.
   """
   if anchor == 'mode':
      pos = np.asarray([grid[np.argmax(d)] for d in density])
   elif anchor == 'mean':
      pos = np.asarray([np.nanmean(s) for s in samples])
   else:
      raise ValueError("bar_anchor must be 'mode' or 'mean'")
   # Interpolate so the leg lands exactly on the plotted curve
   tip = np.asarray([np.interp(p, grid, d) for p, d in zip(pos, density)])
   return pos, tip


# =============================================================================
# Plot parameters
# =============================================================================
# Matplotlib parameters
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
colors = [(166/255, 77/255, 121/255), (67/255, 103/255, 148/255),
   (183/255, 141/255, 48/255)]
    
# Distribution parameters
n_grid = 200  # KDE evaluation points
kde_pad = .05  # grid padding, as a fraction of the pooled data range
n_conditions = 2

# Significance parameters
bar_anchor = 'mode'  # 'mode' (tip of the KDE) or 'mean' (centre of mass)
bar_height = .12  # bracket level above the tallest peak, as a fraction of it
bar_gap = .05  # gap between each tip and its bracket leg
bar_linewidth = 1
text_offset = .02  # asterisk offset above/right of the bracket
head_room = .25  # extra density-axis space for the annotation


# =============================================================================
# Plot the univariate responses for the neural control controlling images
# =============================================================================
# Plot color
color = colors[2]
 
# Get the data
resp_y = resp_high_all.flatten()
resp_y_avg = np.mean(resp_high_all, 1)
base_resp_y_avg = np.mean(base_resp_all)
p_val_y = nc_between_subj_pval
 
# Layout parameters (inches). The x axis is hidden, so the x data units are
# inches too: jitter and strip_pad are physical sizes.
panel_height = 2.5  # Height of the strip and density panels
marg_width = .625  # Width of the density panel
panel_gap = .06  # Gap between the strip and density panels
jitter = .3  # Half-width of the horizontal scatter of the single images
# White space left and right of the strip (keep it above ~.05, otherwise the
# outermost dots get clipped at the panel edges)
strip_pad = .1
fig_margin = 1  # Room around the panels for ticks, labels and asterisks
rng = np.random.default_rng(seed=20200220)
 
# Create the figure. Panel sizes are fixed in inches, so the strip panel is
# exactly as wide as the strip plus its padding.
strip_panel_width = 2*(jitter+strip_pad)
fig_width = 2*fig_margin + strip_panel_width + panel_gap + marg_width
fig_height = 2*fig_margin + panel_height
fig = plt.figure(figsize=(fig_width, fig_height))
ax_strip = fig.add_axes((fig_margin/fig_width, fig_margin/fig_height,
    strip_panel_width/fig_width, panel_height/fig_height))
ax_marg_y = fig.add_axes(((fig_margin+strip_panel_width+panel_gap)/fig_width,
    fig_margin/fig_height, marg_width/fig_width, panel_height/fig_height),
    sharey=ax_strip)
 
# Plot the baseline mean activity
ax_strip.plot([-jitter, jitter], [base_resp_y_avg, base_resp_y_avg], '--k',
    alpha=.25, linewidth=2, zorder=1)
 
# Plot the univariate responses (all images)
x_jitter = rng.uniform(-jitter, jitter, len(resp_y))
ax_strip.scatter(x_jitter, resp_y, c=[color], s=50, alpha=.25,
    edgecolors='k', zorder=2)
 
# Plot the univariate responses (subject average, centered on the strip)
ax_strip.scatter(np.zeros_like(resp_y_avg), resp_y_avg, c=[color],
    s=150, alpha=1, edgecolors='k', zorder=3)
 
# Plot the univariate response distribution
resp_y_finite = resp_y[np.isfinite(resp_y)]
grid_y = np.linspace(np.min(resp_y_finite)-kde_pad*np.ptp(resp_y_finite),
    np.max(resp_y_finite)+kde_pad*np.ptp(resp_y_finite), n_grid)
density_y = marginal_kde(resp_y, grid_y).astype(np.float32)
# Shape: (n_grid,)
ax_marg_y.fill_betweenx(grid_y, 0, density_y, color=color, alpha=.25)
# anchor_positions expects one density row and one response array per
# condition, so pass the single condition as a list of length 1
pos_y, tip_y = anchor_positions(density_y[np.newaxis], grid_y, [resp_y],
    bar_anchor)
 
# Plot the significance
ax_marg_y.text(tip_y[0], pos_y[0], p_to_asterisks(p_val_y), ha='left',
    va='center', color='k', rotation=90)
 
# Marginal axis: density only
max_y = np.max(density_y)
ax_marg_y.set_xlim(0, max_y*(1+head_room))
ax_marg_y.set_axis_off()
 
# x-axis parameters (no x axis: there is a single condition)
ax_strip.set_xlim(-strip_panel_width/2, strip_panel_width/2)
ax_strip.set_xticks([])
ax_strip.spines['bottom'].set_visible(False)
 
# y-axis parameters. bbox_inches='tight' ignores the vertical extent of axis
# labels, so a label longer than the y axis gets cut. Annotations are kept
# whole, so the label is an annotation placed where set_ylabel would put it:
# left of the tick labels, centered on the axis.
ylabel = f'Univariate response\n({tw_all})'
ax_strip.annotate(ylabel, xy=(0, .5),
    xycoords=(lambda renderer: ax_strip.yaxis.get_tightbbox(renderer),
        'axes fraction'),
    xytext=(-plt.rcParams['axes.labelpad'], 0), textcoords='offset points',
    rotation=90, ha='right', va='center', multialignment='center',
    fontsize=fontsize, annotation_clip=False)
yticks = [-1, -0.5, 0, 0.5, 1]
ylabels = [-1, -0.5, 0, 0.5, 1]
# ax_strip.set_yticks(ticks=yticks, labels=ylabels)
bottom_lim = -(max(resp_y.flatten()) / 2)
ax_strip.set_ylim(bottom=bottom_lim)

# Save the figure
file_name = (f'neural_control_roi-{args.roi}_time_window_pair-'
   f'{args.time_window_pair}_imageset-{args.imageset}.svg')
fig.savefig(os.path.join(save_dir, file_name), bbox_inches='tight',
   transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the univariate responses for the RNC controlling images
# =============================================================================
# Get the data
resp_x = [resp_tw_1_h1l2.flatten(), resp_tw_1_l1h2.flatten()]
resp_y = [resp_tw_2_h1l2.flatten(), resp_tw_2_l1h2.flatten()]
resp_x_avg = [np.mean(resp_tw_1_h1l2, 1), np.mean(resp_tw_1_l1h2, 1)]
resp_y_avg = [np.mean(resp_tw_2_h1l2, 1), np.mean(resp_tw_2_l1h2, 1)]
base_resp_x_avg = np.mean(base_resp_1)
base_resp_y_avg = np.mean(base_resp_2)
p_val_x_h1l2 = rnc_between_subj_pval_h1l2_tw_1
p_val_x_l1h2 = rnc_between_subj_pval_l1h2_tw_1
p_val_y_h1l2 = rnc_between_subj_pval_h1l2_tw_2
p_val_y_l1h2 = rnc_between_subj_pval_l1h2_tw_2

# Create the figure
fig = plt.figure(figsize=(4, 4))
gs = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5),
   left=.1, right=.9, bottom=.1, top=.9, wspace=.04, hspace=.04)
ax_joint = fig.add_subplot(gs[1, 0])
ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)
ax_joint.set_box_aspect(1)

# Plot the baseline mean activity
ax_joint.plot([base_resp_x_avg, base_resp_x_avg],
   [min(resp_y[0]), max(resp_y[1])], '--k', alpha=.25, linewidth=2, zorder=1)
ax_joint.plot([min(resp_x[1]), max(resp_x[0])],
   [base_resp_y_avg, base_resp_y_avg], '--k', alpha=.25, linewidth=2,
   zorder=1)

# Plot the univariate responses (all images)
for c in range(n_conditions):
   ax_joint.scatter(resp_x[c], resp_y[c], c=[colors[c]], s=50, alpha=.25,
      edgecolors='k', zorder=2)

# Plot the univariate responses (subject average)
for c in range(n_conditions):
   ax_joint.scatter(resp_x_avg[c], resp_y_avg[c], c=[colors[c]], s=150,
      alpha=1, edgecolors='k', zorder=3)

# Plot the univariate response distributions
all_x = np.concatenate(resp_x)
all_y = np.concatenate(resp_y)
all_x = all_x[np.isfinite(all_x)]
all_y = all_y[np.isfinite(all_y)]
grid_x = np.linspace(np.min(all_x)-kde_pad*np.ptp(all_x),
   np.max(all_x)+kde_pad*np.ptp(all_x), n_grid)
grid_y = np.linspace(np.min(all_y)-kde_pad*np.ptp(all_y),
   np.max(all_y)+kde_pad*np.ptp(all_y), n_grid)
density_x = np.zeros((n_conditions, n_grid), dtype=np.float32)
density_y = np.zeros((n_conditions, n_grid), dtype=np.float32)
for c in range(n_conditions):
   density_x[c] = marginal_kde(resp_x[c], grid_x)
   density_y[c] = marginal_kde(resp_y[c], grid_y)
   # ax_marg_x.plot(grid_x, density_x[c], color=colors[c], linewidth=2)
   ax_marg_x.fill_between(grid_x, 0, density_x[c], color=colors[c], alpha=.25)
   # ax_marg_y.plot(density_y[c], grid_y, color=colors[c], linewidth=2)
   ax_marg_y.fill_betweenx(grid_y, 0, density_y[c], color=colors[c], alpha=.25)
pos_x, tip_x = anchor_positions(density_x, grid_x, resp_x, bar_anchor)
pos_y, tip_y = anchor_positions(density_y, grid_y, resp_y, bar_anchor)

# Plot the significance
ax_marg_x.text(pos_x[0], tip_x[0], p_to_asterisks(p_val_x_h1l2), ha='center',
   va='bottom', color='k')
ax_marg_x.text(pos_x[1], tip_x[1], p_to_asterisks(p_val_x_l1h2), ha='center',
   va='bottom', color='k')
ax_marg_y.text(tip_y[0], pos_y[0], p_to_asterisks(p_val_y_h1l2), ha='left',
   va='center', color='k', rotation=90)
ax_marg_y.text(tip_y[1], pos_y[1], p_to_asterisks(p_val_y_l1h2), ha='left',
   va='center', color='k', rotation=90)

# Drop the spine shared with the joint axis, keep the density axis
max_x = np.max(density_x)
ax_marg_x.tick_params(labelbottom=False, bottom=False)
ax_marg_x.spines['bottom'].set_visible(False)
ax_marg_x.set_ylim(0, max_x*(1+head_room))
max_y = np.max(density_y)
ax_marg_y.tick_params(labelleft=False, left=False)
ax_marg_y.spines['left'].set_visible(False)
ax_marg_y.set_xlim(0, max_y*(1+head_room))
for ax_marg in [ax_marg_x, ax_marg_y]:
   ax_marg.set_axis_off()

# x-axis parameters
xlabel = f'Univariate response\n({tw_1})'
ax_joint.set_xlabel(xlabel, fontsize=fontsize)
xticks = [-2, 0, 2]
xlabels = [-2, 0, 2]
ax_joint.set_xticks(ticks=xticks, labels=xlabels)
# plt.xlim(left=-.75, right=.75)

# y-axis parameters
ylabel = f'Univariate response\n({tw_2})'
ax_joint.set_ylabel(ylabel, fontsize=fontsize)
yticks = [-2, 0, 2]
ylabels = [-2, 0, 2]
ax_joint.set_yticks(ticks=yticks, labels=ylabels)
# plt.ylim(bottom=-1, top=.75)

# Save the figure
file_name = (f'univariate_rnc_roi-{args.roi}_time_window_pair-'
   f'{args.time_window_pair}_imageset-{args.imageset}.svg')
fig.savefig(os.path.join(save_dir, file_name), bbox_inches='tight',
   transparent=True, format='svg')
plt.close()


# # =============================================================================
# # Plot the distributions of image complexity (standard deviation)
# # =============================================================================
# # !!! DELETE !!!
# img_complexity = img_complexity_std
# p_val_complexity = p_val_std

# # Compute the marginal KDEs for the image complexity distributions
# resp_x = [img_complexity['high_1_low_2'],
#    img_complexity['low_1_high_2']]
# all_x = np.concatenate(resp_x)
# all_x = all_x[np.isfinite(all_x)]
# grid_x = np.linspace(np.min(all_x)-kde_pad*np.ptp(all_x),
#    np.max(all_x)+kde_pad*np.ptp(all_x), n_grid)
# density_x = np.zeros((n_conditions, n_grid), dtype=np.float32)
# for c, key in enumerate(['high_1_low_2', 'low_1_high_2']):
#    density_x[c] = marginal_kde(img_complexity[key], grid_x)
# pos_x, tip_x = anchor_positions(density_x, grid_x, resp_x, bar_anchor)

# # Create the figure
# fig = plt.figure(figsize=(10, 10))

# # Plot the image complexity distributions
# for c, key in enumerate(['high_1_low_2', 'low_1_high_2']):
#    plt.plot(grid_x, density_x[c], color=colors[c], linewidth=2)
#    plt.fill_between(grid_x, 0, density_x[c], color=colors[c], alpha=.25)

# # Plot the significance # !!!
# max_x = np.max(density_x)
# bar_level_x = max_x * (1 + bar_height)
# plt.plot([pos_x[0], pos_x[0], pos_x[1], pos_x[1]],
#    [tip_x[0]+bar_gap*max_x, bar_level_x, bar_level_x, tip_x[1]+bar_gap*max_x],
#    color='k', linewidth=bar_linewidth, solid_capstyle='butt',
#    label='_nolegend_')
# plt.text(np.mean(pos_x), bar_level_x+text_offset*max_x,
#    p_to_asterisks(p_val_complexity), ha='center', va='bottom', color='k')

# # x-axis parameters
# xlabel = f'Image complexity'
# plt.xlabel(xlabel, fontsize=fontsize)
# xticks = [1, -0.5, 0, 0.5, 1]
# xlabels = [1, -0.5, 0, 0.5, 1]
# # plt.xticks(ticks=xticks, labels=xlabels)
# # plt.xlim(left=-.75, right=.75)

# # y-axis parameters
# ylabel = f'Frequency (a.u.)'
# plt.ylabel(ylabel, fontsize=fontsize)
# yticks = [1, -0.5, 0, 0.5, 1]
# ylabels = [1, -0.5, 0, 0.5, 1]
# # plt.yticks(ticks=yticks, labels=ylabels)
# # plt.ylim(bottom=-1, top=.75)

# # Save the figure
# file_name = (f'image_complexity_std_roi-{args.roi}_time_window_pair-'
#    f'{args.time_window_pair}_imageset-{args.imageset}.svg')
# fig.savefig(os.path.join(save_dir, file_name), bbox_inches='tight',
#    transparent=True, format='svg')
# plt.close()


# # =============================================================================
# # Plot the distributions of image complexity (best DNN layer)
# # =============================================================================
# # !!! DELETE !!!
# img_complexity = img_complexity_best_layer
# p_val_complexity = p_val_best_layer

# # Compute the marginal KDEs for the image complexity distributions
# resp_x = [img_complexity['high_1_low_2'],
#    img_complexity['low_1_high_2']]
# all_x = np.concatenate(resp_x)
# all_x = all_x[np.isfinite(all_x)]
# grid_x = np.linspace(np.min(all_x)-kde_pad*np.ptp(all_x),
#    np.max(all_x)+kde_pad*np.ptp(all_x), n_grid)
# density_x = np.zeros((n_conditions, n_grid), dtype=np.float32)
# for c, key in enumerate(['high_1_low_2', 'low_1_high_2']):
#    density_x[c] = marginal_kde(img_complexity[key], grid_x)
# pos_x, tip_x = anchor_positions(density_x, grid_x, resp_x, bar_anchor)

# # Create the figure
# fig = plt.figure(figsize=(10, 10))

# # Plot the image complexity distributions
# for c, key in enumerate(['high_1_low_2', 'low_1_high_2']):
#    plt.plot(grid_x, density_x[c], color=colors[c], linewidth=2)
#    plt.fill_between(grid_x, 0, density_x[c], color=colors[c], alpha=.25)

# # Plot the significance # !!!
# max_x = np.max(density_x)
# bar_level_x = max_x * (1 + bar_height)
# plt.plot([pos_x[0], pos_x[0], pos_x[1], pos_x[1]],
#    [tip_x[0]+bar_gap*max_x, bar_level_x, bar_level_x, tip_x[1]+bar_gap*max_x],
#    color='k', linewidth=bar_linewidth, solid_capstyle='butt',
#    label='_nolegend_')
# plt.text(np.mean(pos_x), bar_level_x+text_offset*max_x,
#    p_to_asterisks(p_val_complexity), ha='center', va='bottom', color='k')

# # x-axis parameters
# xlabel = f'Image complexity'
# plt.xlabel(xlabel, fontsize=fontsize)
# xticks = [1, -0.5, 0, 0.5, 1]
# xlabels = [1, -0.5, 0, 0.5, 1]
# # plt.xticks(ticks=xticks, labels=xlabels)
# # plt.xlim(left=-.75, right=.75)

# # y-axis parameters
# ylabel = f'Frequency (a.u.)'
# plt.ylabel(ylabel, fontsize=fontsize)
# yticks = [1, -0.5, 0, 0.5, 1]
# ylabels = [1, -0.5, 0, 0.5, 1]
# # plt.yticks(ticks=yticks, labels=ylabels)
# # plt.ylim(bottom=-1, top=.75)

# # Save the figure
# file_name = (f'image_complexity_best_layer_roi-{args.roi}_time_window_pair-'
#    f'{args.time_window_pair}_imageset-{args.imageset}.svg')
# fig.savefig(os.path.join(save_dir, file_name), bbox_inches='tight',
#    transparent=True, format='svg')
# plt.close()


# # =============================================================================
# # Plot the distributions of image complexity (PNG)
# # =============================================================================
# # !!! DELETE !!!
# img_complexity = img_complexity_png
# p_val_complexity = p_val_png

# # Compute the marginal KDEs for the image complexity distributions
# resp_x = [img_complexity['high_1_low_2'],
#    img_complexity['low_1_high_2']]
# all_x = np.concatenate(resp_x)
# all_x = all_x[np.isfinite(all_x)]
# grid_x = np.linspace(np.min(all_x)-kde_pad*np.ptp(all_x),
#    np.max(all_x)+kde_pad*np.ptp(all_x), n_grid)
# density_x = np.zeros((n_conditions, n_grid), dtype=np.float32)
# for c, key in enumerate(['high_1_low_2', 'low_1_high_2']):
#    density_x[c] = marginal_kde(img_complexity[key], grid_x)
# pos_x, tip_x = anchor_positions(density_x, grid_x, resp_x, bar_anchor)

# # Create the figure
# fig = plt.figure(figsize=(10, 10))

# # Plot the image complexity distributions
# for c, key in enumerate(['high_1_low_2', 'low_1_high_2']):
#    plt.plot(grid_x, density_x[c], color=colors[c], linewidth=2)
#    plt.fill_between(grid_x, 0, density_x[c], color=colors[c], alpha=.25)

# # Plot the significance # !!!
# max_x = np.max(density_x)
# bar_level_x = max_x * (1 + bar_height)
# plt.plot([pos_x[0], pos_x[0], pos_x[1], pos_x[1]],
#    [tip_x[0]+bar_gap*max_x, bar_level_x, bar_level_x, tip_x[1]+bar_gap*max_x],
#    color='k', linewidth=bar_linewidth, solid_capstyle='butt',
#    label='_nolegend_')
# plt.text(np.mean(pos_x), bar_level_x+text_offset*max_x,
#    p_to_asterisks(p_val_complexity), ha='center', va='bottom', color='k')

# # x-axis parameters
# xlabel = f'Image complexity'
# plt.xlabel(xlabel, fontsize=fontsize)
# xticks = [1, -0.5, 0, 0.5, 1]
# xlabels = [1, -0.5, 0, 0.5, 1]
# # plt.xticks(ticks=xticks, labels=xlabels)
# # plt.xlim(left=-.75, right=.75)

# # y-axis parameters
# ylabel = f'Frequency (a.u.)'
# plt.ylabel(ylabel, fontsize=fontsize)
# yticks = [1, -0.5, 0, 0.5, 1]
# ylabels = [1, -0.5, 0, 0.5, 1]
# # plt.yticks(ticks=yticks, labels=ylabels)
# # plt.ylim(bottom=-1, top=.75)

# # Save the figure
# file_name = (f'image_complexity_png_roi-{args.roi}_time_window_pair-'
#    f'{args.time_window_pair}_imageset-{args.imageset}.svg')
# fig.savefig(os.path.join(save_dir, file_name), bbox_inches='tight',
#    transparent=True, format='svg')
# plt.close()


# # =============================================================================
# # Plot the distributions of image complexity (JPG)
# # =============================================================================
# # !!! DELETE !!!
# img_complexity = img_complexity_jpg
# p_val_complexity = p_val_jpg

# # Compute the marginal KDEs for the image complexity distributions
# resp_x = [img_complexity['high_1_low_2'],
#    img_complexity['low_1_high_2']]
# all_x = np.concatenate(resp_x)
# all_x = all_x[np.isfinite(all_x)]
# grid_x = np.linspace(np.min(all_x)-kde_pad*np.ptp(all_x),
#    np.max(all_x)+kde_pad*np.ptp(all_x), n_grid)
# density_x = np.zeros((n_conditions, n_grid), dtype=np.float32)
# for c, key in enumerate(['high_1_low_2', 'low_1_high_2']):
#    density_x[c] = marginal_kde(img_complexity[key], grid_x)
# pos_x, tip_x = anchor_positions(density_x, grid_x, resp_x, bar_anchor)

# # Create the figure
# fig = plt.figure(figsize=(10, 10))

# # Plot the image complexity distributions
# for c, key in enumerate(['high_1_low_2', 'low_1_high_2']):
#    plt.plot(grid_x, density_x[c], color=colors[c], linewidth=2)
#    plt.fill_between(grid_x, 0, density_x[c], color=colors[c], alpha=.25)

# # Plot the significance # !!!
# max_x = np.max(density_x)
# bar_level_x = max_x * (1 + bar_height)
# plt.plot([pos_x[0], pos_x[0], pos_x[1], pos_x[1]],
#    [tip_x[0]+bar_gap*max_x, bar_level_x, bar_level_x, tip_x[1]+bar_gap*max_x],
#    color='k', linewidth=bar_linewidth, solid_capstyle='butt',
#    label='_nolegend_')
# plt.text(np.mean(pos_x), bar_level_x+text_offset*max_x,
#    p_to_asterisks(p_val_complexity), ha='center', va='bottom', color='k')

# # x-axis parameters
# xlabel = f'Image complexity'
# plt.xlabel(xlabel, fontsize=fontsize)
# xticks = [1, -0.5, 0, 0.5, 1]
# xlabels = [1, -0.5, 0, 0.5, 1]
# # plt.xticks(ticks=xticks, labels=xlabels)
# # plt.xlim(left=-.75, right=.75)

# # y-axis parameters
# ylabel = f'Frequency (a.u.)'
# plt.ylabel(ylabel, fontsize=fontsize)
# yticks = [1, -0.5, 0, 0.5, 1]
# ylabels = [1, -0.5, 0, 0.5, 1]
# # plt.yticks(ticks=yticks, labels=ylabels)
# # plt.ylim(bottom=-1, top=.75)

# # Save the figure
# file_name = (f'image_complexity_jpg_roi-{args.roi}_time_window_pair-'
#    f'{args.time_window_pair}_imageset-{args.imageset}.svg')
# fig.savefig(os.path.join(save_dir, file_name), bbox_inches='tight',
#    transparent=True, format='svg')
# plt.close()


# # =============================================================================
# # Plot the distributions of image complexity (ICNET)
# # =============================================================================
# # !!! DELETE !!!
# img_complexity = img_complexity_icnet
# p_val_complexity = p_val_icnet

# # Compute the marginal KDEs for the image complexity distributions
# resp_x = [img_complexity['high_1_low_2'],
#    img_complexity['low_1_high_2']]
# all_x = np.concatenate(resp_x)
# all_x = all_x[np.isfinite(all_x)]
# grid_x = np.linspace(np.min(all_x)-kde_pad*np.ptp(all_x),
#    np.max(all_x)+kde_pad*np.ptp(all_x), n_grid)
# density_x = np.zeros((n_conditions, n_grid), dtype=np.float32)
# for c, key in enumerate(['high_1_low_2', 'low_1_high_2']):
#    density_x[c] = marginal_kde(img_complexity[key], grid_x)
# pos_x, tip_x = anchor_positions(density_x, grid_x, resp_x, bar_anchor)

# # Create the figure
# fig = plt.figure(figsize=(10, 10))

# # Plot the image complexity distributions
# for c, key in enumerate(['high_1_low_2', 'low_1_high_2']):
#    plt.plot(grid_x, density_x[c], color=colors[c], linewidth=2)
#    plt.fill_between(grid_x, 0, density_x[c], color=colors[c], alpha=.25)

# # Plot the significance # !!!
# max_x = np.max(density_x)
# bar_level_x = max_x * (1 + bar_height)
# plt.plot([pos_x[0], pos_x[0], pos_x[1], pos_x[1]],
#    [tip_x[0]+bar_gap*max_x, bar_level_x, bar_level_x, tip_x[1]+bar_gap*max_x],
#    color='k', linewidth=bar_linewidth, solid_capstyle='butt',
#    label='_nolegend_')
# plt.text(np.mean(pos_x), bar_level_x+text_offset*max_x,
#    p_to_asterisks(p_val_complexity), ha='center', va='bottom', color='k')

# # x-axis parameters
# xlabel = f'Image complexity'
# plt.xlabel(xlabel, fontsize=fontsize)
# xticks = [1, -0.5, 0, 0.5, 1]
# xlabels = [1, -0.5, 0, 0.5, 1]
# # plt.xticks(ticks=xticks, labels=xlabels)
# # plt.xlim(left=-.75, right=.75)

# # y-axis parameters
# ylabel = f'Frequency (a.u.)'
# plt.ylabel(ylabel, fontsize=fontsize)
# yticks = [1, -0.5, 0, 0.5, 1]
# ylabels = [1, -0.5, 0, 0.5, 1]
# # plt.yticks(ticks=yticks, labels=ylabels)
# # plt.ylim(bottom=-1, top=.75)

# # Save the figure
# file_name = (f'image_complexity_icnet_roi-{args.roi}_time_window_pair-'
#    f'{args.time_window_pair}_imageset-{args.imageset}.svg')
# fig.savefig(os.path.join(save_dir, file_name), bbox_inches='tight',
#    transparent=True, format='svg')
# plt.close()


# # =============================================================================
# # Plot the distributions of image complexity (nagle_lavie)
# # =============================================================================
# # !!! DELETE !!!
# img_complexity = img_complexity_nagle_lavie
# p_val_complexity = p_val_nagle_lavie

# # Compute the marginal KDEs for the image complexity distributions
# resp_x = [img_complexity['high_1_low_2'],
#    img_complexity['low_1_high_2']]
# all_x = np.concatenate(resp_x)
# all_x = all_x[np.isfinite(all_x)]
# grid_x = np.linspace(np.min(all_x)-kde_pad*np.ptp(all_x),
#    np.max(all_x)+kde_pad*np.ptp(all_x), n_grid)
# density_x = np.zeros((n_conditions, n_grid), dtype=np.float32)
# for c, key in enumerate(['high_1_low_2', 'low_1_high_2']):
#    density_x[c] = marginal_kde(img_complexity[key], grid_x)
# pos_x, tip_x = anchor_positions(density_x, grid_x, resp_x, bar_anchor)

# # Create the figure
# fig = plt.figure(figsize=(10, 10))

# # Plot the image complexity distributions
# for c, key in enumerate(['high_1_low_2', 'low_1_high_2']):
#    plt.plot(grid_x, density_x[c], color=colors[c], linewidth=2)
#    plt.fill_between(grid_x, 0, density_x[c], color=colors[c], alpha=.25)

# # Plot the significance # !!!
# max_x = np.max(density_x)
# bar_level_x = max_x * (1 + bar_height)
# plt.plot([pos_x[0], pos_x[0], pos_x[1], pos_x[1]],
#    [tip_x[0]+bar_gap*max_x, bar_level_x, bar_level_x, tip_x[1]+bar_gap*max_x],
#    color='k', linewidth=bar_linewidth, solid_capstyle='butt',
#    label='_nolegend_')
# plt.text(np.mean(pos_x), bar_level_x+text_offset*max_x,
#    p_to_asterisks(p_val_complexity), ha='center', va='bottom', color='k')

# # x-axis parameters
# xlabel = f'Image complexity'
# plt.xlabel(xlabel, fontsize=fontsize)
# xticks = [1, -0.5, 0, 0.5, 1]
# xlabels = [1, -0.5, 0, 0.5, 1]
# # plt.xticks(ticks=xticks, labels=xlabels)
# # plt.xlim(left=-.75, right=.75)

# # y-axis parameters
# ylabel = f'Frequency (a.u.)'
# plt.ylabel(ylabel, fontsize=fontsize)
# yticks = [1, -0.5, 0, 0.5, 1]
# ylabels = [1, -0.5, 0, 0.5, 1]
# # plt.yticks(ticks=yticks, labels=ylabels)
# # plt.ylim(bottom=-1, top=.75)

# # Save the figure
# file_name = (f'image_complexity_nagle_lavie_roi-{args.roi}_time_window_pair-'
#    f'{args.time_window_pair}_imageset-{args.imageset}.svg')
# fig.savefig(os.path.join(save_dir, file_name), bbox_inches='tight',
#    transparent=True, format='svg')
# plt.close()