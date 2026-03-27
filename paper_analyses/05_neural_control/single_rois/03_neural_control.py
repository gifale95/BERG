"""Apply neural control to find images that drive or suppress the in silico
monkey electrophysiology responses.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico 
    responses.
subject : int
    Subject identifier for the monkey encoding model. Since the used encoding
    models are trained on the TVSD data, valid subject identifiers are "N" and
    "F".
roi: str
    ROI used. Valid values are "V1", "V4", and "IT".
control: str
    If "early-drive_late-drive", then both the early (25-100ms) and late
    (101-200ms) part of the epoch are driven.
    If "early-suppress_late-suppress", then both the early and late part of the
    epoch are suppressed.
    If "early-drive_late-suppress", then the early part of the epoch is driven
    while the late part is suppressed.
    If "early-suppress_late-drive", then the early part of the epoch is
    suppressed while the late part is driven.
n_images: int
    Number of retained controlling or baseline images.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subject', default='N', type=str)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--control', default='early-drive_late-drive', type=str)
parser.add_argument('--n_images', default=50, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Neural control <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the in silico neural responses for the ~1.3M ILSVRC-2012 train images
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'neural_control', 'insilico_responses',
args.encoding_model)
file_name = f'insilico_responses_sub-{args.subject}_roi-{args.roi}.npy'

data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
insilico_resp = data['responses']
metadata = data['metadata']

# Average the in silico neural responses across the time window around peak
# activity (as in the TVSD paper)
# times = metadata[0]['utah_array']['times']
# peaks = {
#     'V1': (25, 125),
#     'V4': (50, 150),
#     'IT': (75, 175)
# }
# t_min = np.where(times == peaks[args.roi][0])[0][0]
# t_max = np.where(times == peaks[args.roi][1])[0][0]

# Average the in silico neural responses across early parts of the epoch
# (25-100ms), late parts of the epoch (101-200ms), or the entire epoch
# (25-200ms)
times = metadata['utah_array']['times']
t_min_early = np.where(times == 25)[0][0]
t_max_early = np.where(times == 100)[0][0]
t_min_late = np.where(times == 101)[0][0]
t_max_late = np.where(times == 199)[0][0]
insilico_resp_early = np.mean(insilico_resp[:,t_min_early:t_max_early+1], 1)
insilico_resp_late = np.mean(insilico_resp[:,t_min_late:t_max_late+1], 1)


# =============================================================================
# Load the baseline results
# =============================================================================
# Load the baseline results
data_dir = os.path.join(args.berg_dir, 'neural_control', 'single_rois',
    'quantitative_results', args.encoding_model,
    f'sub-{args.subject}_roi-{args.roi}_baseline.npy')
baseline_results = np.load(data_dir, allow_pickle=True).item()

# Average the baseline responses across early parts of the epoch
# (25-100ms), late parts of the epoch (101-200ms), or the entire epoch
# (25-200ms)
baseline_resp = np.mean(baseline_results['baseline_resp'], 0)
baseline_resp_early = np.mean(baseline_resp[t_min_early:t_max_early+1])
baseline_resp_late = np.mean(baseline_resp[t_min_late:t_max_late+1])
baseline_resp_full = np.mean(baseline_resp[t_min_early:t_max_late+1])


# =============================================================================
# Neural control
# =============================================================================
# Response score margin used to constrain the selection of the controlling
# images
margin = 2

# Select the top N images that drive or suppress both early and late part of
# the epoch
if args.control in ['early-drive_late-drive', 'early-suppress_late-suppress']:

    response_sum = insilico_resp_early + insilico_resp_late

    # Select the top N images that drive both the early and late part of the
    # epoch
    if args.control == 'early-drive_late-drive':
        img_control = np.argsort(response_sum)[::-1].astype(np.float32)
        # Ignore image conditions with responses below the baseline scores
        # (plus a margin)
        idx_bad_early = np.where(
            insilico_resp_early[img_control.astype(np.int32)] < \
            baseline_resp_early+margin)[0]
        idx_bad_late = np.where(
            insilico_resp_late[img_control.astype(np.int32)] < \
            baseline_resp_late+margin)[0]
        img_control[idx_bad_early] = np.nan
        img_control[idx_bad_late] = np.nan

    # Select the top N images that suppress both the early and late part of the
    # epoch
    elif args.control == 'early-suppress_late-suppress':
        img_control = np.argsort(response_sum).astype(np.float32)
        # Ignore image conditions with responses above the baseline scores
        # (plus a margin)
        idx_bad_early = np.where(
            insilico_resp_early[img_control.astype(np.int32)] > \
            baseline_resp_early-margin)[0]
        idx_bad_late = np.where(
            insilico_resp_late[img_control.astype(np.int32)] > \
            baseline_resp_late-margin)[0]
        img_control[idx_bad_early] = np.nan
        img_control[idx_bad_late] = np.nan

# Select the top N images that drive the early while suppressing the late part
# of the epoch, or vice versa
elif args.control in ['early-drive_late-suppress', 'early-suppress_late-drive']:

    response_diff = insilico_resp_early - insilico_resp_late

    # Select the top N images that drive the early while suppressing the late
    # part of the epoch
    if args.control == 'early-drive_late-suppress':
        img_control = np.argsort(response_diff)[::-1].astype(np.float32)
        # Ignore image conditions with responses below (for early time points)
        # or above (for late time points) the baseline scores (plus a margin)
        idx_bad_early = np.where(
            insilico_resp_early[img_control.astype(np.int32)] < \
            baseline_resp_early+margin)[0]
        idx_bad_late = np.where(
            insilico_resp_late[img_control.astype(np.int32)] > \
            baseline_resp_late-margin)[0]
        img_control[idx_bad_early] = np.nan
        img_control[idx_bad_late] = np.nan

    # Select the top N images that suppress the early while driving the late
    # part of the epoch
    elif args.control == 'early-suppress_late-drive':
        img_control = np.argsort(response_diff).astype(np.float32)
        # Ignore image conditions with responses above (for early time points)
        # or below (for late time points) the baseline scores (plus a margin)
        idx_bad_early = np.where(
            insilico_resp_early[img_control.astype(np.int32)] > \
            baseline_resp_early-margin)[0]
        idx_bad_late = np.where(
            insilico_resp_late[img_control.astype(np.int32)] < \
            baseline_resp_late+margin)[0]
        img_control[idx_bad_early] = np.nan
        img_control[idx_bad_late] = np.nan


# =============================================================================
# Save the quantitative neural control results
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_control', 'single_rois',
    'quantitative_results', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = f'sub-{args.subject}_roi-{args.roi}_{args.control}.npy'

np.save(os.path.join(save_dir, file_name), img_control)