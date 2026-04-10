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
roi_1: str
    First ROI used. Valid values are "V1", "V4", and "IT".
roi_2: str
    Second ROI used. Valid values are "V1", "V4", and "IT". If None, then only
    one ROI (roi_1) is used for neural control.
control_roi_1: str
    Neural control objective for the first ROI.
    If "early-drive_late-drive", then both the early (25-100ms) and late
    (101-200ms) part of the epoch are driven.
    If "early-suppress_late-suppress", then both the early and late part of the
    epoch are suppressed.
    If "early-drive_late-suppress", then the early part of the epoch is driven
    while the late part is suppressed.
    If "early-suppress_late-drive", then the early part of the epoch is
    suppressed while the late part is driven.
control_roi_2: str
    Neural control objective for the second ROI. The valid values are the same
    as for control_roi_1.
n_images: int
    Number of retained controlling or baseline images.
margin: int
    Response score margin used to constrain the selection of the controlling
    images. For example, if margin=2, then only images with in silico responses
    that are at least 2 points above (for driving) or below (for suppressing)
    the baseline scores are retained as controlling images.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subject', default='N', type=str)
parser.add_argument('--roi_1', default='V1', type=str)
parser.add_argument('--roi_2', default=None, type=str)
parser.add_argument('--control_roi_1', default='early-drive_late-drive', type=str)
parser.add_argument('--control_roi_2', default=None, type=str)
parser.add_argument('--n_images', default=100, type=int)
parser.add_argument('--margin', default=0, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Neural control <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the in silico neural responses for the ~1.3M ILSVRC-2012 train images
# =============================================================================
data_dir_roi_1 = os.path.join(args.berg_dir, 'neural_control',
    'neural_control', 'insilico_responses', args.encoding_model,
    f'insilico_responses_sub-{args.subject}_roi-{args.roi_1}.npy')
data_roi_1 = np.load(data_dir_roi_1, allow_pickle=True).item()
resp_roi_1 = data_roi_1['responses']
metadata = data_roi_1['metadata']
del data_roi_1

if args.roi_2 is not None:
    data_dir_roi_2 = os.path.join(args.berg_dir, 'neural_control',
        'neural_control', 'insilico_responses', args.encoding_model,
        f'insilico_responses_sub-{args.subject}_roi-{args.roi_2}.npy')
    resp_roi_2 = np.load(data_dir_roi_2, allow_pickle=True).item()['responses']


# =============================================================================
# Load the baseline results
# =============================================================================
data_dir_roi_1 = os.path.join(args.berg_dir, 'neural_control',
    'neural_control', 'quantitative_results', args.encoding_model,
    f'sub-{args.subject}_roi-{args.roi_1}_baseline.npy')
base_roi_1 = np.mean(np.load(data_dir_roi_1,allow_pickle=True).item()\
    ['baseline_resp'], 0)

if args.roi_2 is not None:
    data_dir_roi_2 = os.path.join(args.berg_dir, 'neural_control',
        'neural_control', 'quantitative_results', args.encoding_model,
        f'sub-{args.subject}_roi-{args.roi_2}_baseline.npy')
    base_roi_2 = np.mean(np.load(data_dir_roi_2, allow_pickle=True).item()\
        ['baseline_resp'], 0)


# =============================================================================
# Average the responses across early (25-100ms) or late (101-200ms) time points
# =============================================================================
times = metadata['utah_array']['times']
t_min_early = np.where(times == 25)[0][0]
t_max_early = np.where(times == 100)[0][0]
t_min_late = np.where(times == 101)[0][0]
t_max_late = np.where(times == 199)[0][0]

resp_roi_1_early = np.mean(resp_roi_1[:,t_min_early:t_max_early+1], 1)
resp_roi_1_late = np.mean(resp_roi_1[:,t_min_late:t_max_late+1], 1)
base_roi_1_early = np.mean(base_roi_1[t_min_early:t_max_early+1])
base_roi_1_late = np.mean(base_roi_1[t_min_late:t_max_late+1])

if args.roi_2 is not None:
    resp_roi_2_early = np.mean(resp_roi_2[:,t_min_early:t_max_early+1], 1)
    resp_roi_2_late = np.mean(resp_roi_2[:,t_min_late:t_max_late+1], 1)
    base_roi_2_early = np.mean(base_roi_2[t_min_early:t_max_early+1])
    base_roi_2_late = np.mean(base_roi_2[t_min_late:t_max_late+1])


# =============================================================================
# Neural control
# =============================================================================
# Select the top N images that drive or suppress both early and late part of
# the epoch
if args.control_roi_1 in ['early-drive_late-drive', 'early-suppress_late-suppress']:

    response_sum = resp_roi_1_early + resp_roi_1_late

    # Select the top N images that drive both the early and late part of the
    # epoch
    if args.control_roi_1 == 'early-drive_late-drive':
        img_control_roi_1 = np.argsort(response_sum)[::-1].astype(np.float32)
        # Ignore image conditions with responses below the baseline scores
        # (plus a margin)
        idx_bad_early = np.where(
            resp_roi_1_early[img_control_roi_1.astype(np.int32)] < \
            base_roi_1_early+args.margin)[0]
        idx_bad_late = np.where(
            resp_roi_1_late[img_control_roi_1.astype(np.int32)] < \
            base_roi_1_late+args.margin)[0]
        img_control_roi_1[idx_bad_early] = np.nan
        img_control_roi_1[idx_bad_late] = np.nan

    # Select the top N images that suppress both the early and late part of the
    # epoch
    elif args.control_roi_1 == 'early-suppress_late-suppress':
        img_control_roi_1 = np.argsort(response_sum).astype(np.float32)
        # Ignore image conditions with responses above the baseline scores
        # (plus a margin)
        idx_bad_early = np.where(
            resp_roi_1_early[img_control_roi_1.astype(np.int32)] > \
            base_roi_1_early-args.margin)[0]
        idx_bad_late = np.where(
            resp_roi_1_late[img_control_roi_1.astype(np.int32)] > \
            base_roi_1_late-args.margin)[0]
        img_control_roi_1[idx_bad_early] = np.nan
        img_control_roi_1[idx_bad_late] = np.nan

# Select the top N images that drive the early while suppressing the late part
# of the epoch, or vice versa
elif args.control_roi_1 in ['early-drive_late-suppress', 'early-suppress_late-drive']:

    response_diff = resp_roi_1_early - resp_roi_1_late

    # Select the top N images that drive the early while suppressing the late
    # part of the epoch
    if args.control_roi_1 == 'early-drive_late-suppress':
        img_control_roi_1 = np.argsort(response_diff)[::-1].astype(np.float32)
        # Ignore image conditions with responses below (for early time points)
        # or above (for late time points) the baseline scores (plus a margin)
        idx_bad_early = np.where(
            resp_roi_1_early[img_control_roi_1.astype(np.int32)] < \
            base_roi_1_early+args.margin)[0]
        idx_bad_late = np.where(
            resp_roi_1_late[img_control_roi_1.astype(np.int32)] > \
            base_roi_1_late-args.margin)[0]
        img_control_roi_1[idx_bad_early] = np.nan
        img_control_roi_1[idx_bad_late] = np.nan

    # Select the top N images that suppress the early while driving the late
    # part of the epoch
    elif args.control_roi_1 == 'early-suppress_late-drive':
        img_control_roi_1 = np.argsort(response_diff).astype(np.float32)
        # Ignore image conditions with responses above (for early time points)
        # or below (for late time points) the baseline scores (plus a margin)
        idx_bad_early = np.where(
            resp_roi_1_early[img_control_roi_1.astype(np.int32)] > \
            base_roi_1_early-args.margin)[0]
        idx_bad_late = np.where(
            resp_roi_1_late[img_control_roi_1.astype(np.int32)] < \
            base_roi_1_late+args.margin)[0]
        img_control_roi_1[idx_bad_early] = np.nan
        img_control_roi_1[idx_bad_late] = np.nan


# =============================================================================
# Neural control (ROI 2)
# =============================================================================
if args.roi_2 is not None:

    # Select the top N images that drive or suppress both early and late part
    # of the epoch
    if args.control_roi_2 in ['early-drive_late-drive', 'early-suppress_late-suppress']:

        response_sum = resp_roi_2_early + resp_roi_2_late

        # Select the top N images that drive both the early and late part of
        # the epoch
        if args.control_roi_2 == 'early-drive_late-drive':
            img_control_roi_2 = np.argsort(response_sum)[::-1].astype(
                np.float32)
            # Ignore image conditions with responses below the baseline scores
            # (plus a margin)
            idx_bad_early = np.where(
                resp_roi_2_early[img_control_roi_2.astype(np.int32)] < \
                base_roi_2_early+args.margin)[0]
            idx_bad_late = np.where(
                resp_roi_2_late[img_control_roi_2.astype(np.int32)] < \
                base_roi_2_late+args.margin)[0]
            img_control_roi_2[idx_bad_early] = np.nan
            img_control_roi_2[idx_bad_late] = np.nan

        # Select the top N images that suppress both the early and late part of
        # the epoch
        elif args.control_roi_2 == 'early-suppress_late-suppress':
            img_control_roi_2 = np.argsort(response_sum).astype(np.float32)
            # Ignore image conditions with responses above the baseline scores
            # (plus a margin)
            idx_bad_early = np.where(
                resp_roi_2_early[img_control_roi_2.astype(np.int32)] > \
                base_roi_2_early-args.margin)[0]
            idx_bad_late = np.where(
                resp_roi_2_late[img_control_roi_2.astype(np.int32)] > \
                base_roi_2_late-args.margin)[0]
            img_control_roi_2[idx_bad_early] = np.nan
            img_control_roi_2[idx_bad_late] = np.nan

    # Select the top N images that drive the early while suppressing the late
    # part of the epoch, or vice versa
    elif args.control_roi_2 in ['early-drive_late-suppress', 'early-suppress_late-drive']:

        response_diff = resp_roi_2_early - resp_roi_2_late

        # Select the top N images that drive the early while suppressing the
        # late part of the epoch
        if args.control_roi_2 == 'early-drive_late-suppress':
            img_control_roi_2 = np.argsort(response_diff)[::-1].astype(
                np.float32)
            # Ignore image conditions with responses below (for early time
            # points) or above (for late time points) the baseline scores (plus
            # a margin)
            idx_bad_early = np.where(
                resp_roi_2_early[img_control_roi_2.astype(np.int32)] < \
                base_roi_2_early+args.margin)[0]
            idx_bad_late = np.where(
                resp_roi_2_late[img_control_roi_2.astype(np.int32)] > \
                base_roi_2_late-args.margin)[0]
            img_control_roi_2[idx_bad_early] = np.nan
            img_control_roi_2[idx_bad_late] = np.nan

        # Select the top N images that suppress the early while driving the
        # late part of the epoch
        elif args.control_roi_2 == 'early-suppress_late-drive':
            img_control_roi_2 = np.argsort(response_diff).astype(np.float32)
            # Ignore image conditions with responses above (for early time
            # points) or below (for late time points) the baseline scores (plus
            # a margin)
            idx_bad_early = np.where(
                resp_roi_2_early[img_control_roi_2.astype(np.int32)] > \
                base_roi_2_early-args.margin)[0]
            idx_bad_late = np.where(
                resp_roi_2_late[img_control_roi_2.astype(np.int32)] < \
                base_roi_2_late+args.margin)[0]
            img_control_roi_2[idx_bad_early] = np.nan
            img_control_roi_2[idx_bad_late] = np.nan


# =============================================================================
# Combine the neural control scores for both ROIs
# =============================================================================
# If only one ROI is used, then keep the top N images based on the control
# scores for that ROI
if args.roi_2 is None:
    img_control = img_control_roi_1[~np.isnan(img_control_roi_1)]\
        [:args.n_images].astype(int)

# If two ROIs are used, then keep the images that are in the top N for both
# ROIs (ranked by the sum of the ranks for both ROIs)
else:
    # Remove NaNs
    img_roi_1 = img_control_roi_1[~np.isnan(img_control_roi_1)].astype(int)
    img_roi_2 = img_control_roi_2[~np.isnan(img_control_roi_2)].astype(int)
    # Get first occurrence indices
    pos_1 = {}
    for i, val in enumerate(img_roi_1):
        pos_1[val] = i
    pos_2 = {}
    for i, val in enumerate(img_roi_2):
        pos_2[val] = i
    # Find common elements
    common = set(pos_1.keys()) & set(pos_2.keys())
    # Rank by combined position (lower = earlier in both)
    ranked = sorted(common, key=lambda x: pos_1[x] + pos_2[x])
    # Keep the best N image conditions
    img_control = np.array(ranked[:args.n_images])

# Throw an error if less than the specified number of images meet the control
# criteria
if len(img_control) < args.n_images:
    raise ValueError("Less than the specified number of images meet the neural control criteria.")


# =============================================================================
# Save the quantitative neural control results
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'neural_control', 'neural_control',
    'quantitative_results', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

if args.roi_2 is not None:
    file_name = (f'sub-{args.subject}_roi_1-{args.roi_1}_{args.control_roi_1}'
        f'_roi_2-{args.roi_2}_{args.control_roi_2}.npy')
else:
    file_name = f'sub-{args.subject}_roi-{args.roi_1}_{args.control_roi_1}.npy'

np.save(os.path.join(save_dir, file_name), img_control)