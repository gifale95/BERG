"""Create the in silico EEG RSMs that will be later used by the multivariate
RNC algorithm. Each RSM consists in the pairwise comparisons for 10k
ILSVRC-2012 validation images (the first 10 images of each category),
for the chosen EEG channels and time point.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico EEG
    responses.
subject : int
    The subject identifier for the EEG encoding models. Since the used
    encoidng models are trained on THINGS EEG2 data, valid subject identifiers
    are integers from 1 to 10.
channels : string
    String containing the EEG channel type(s) retained for the analyses,
    separated by a comma. Possible values are: 'O' (occipital), 'P'
    (posterior), 'T' (temporal), 'C' (central), 'F' (frontal). Alternatively,
    the list can also contain the names of the individual channels used.
time : float
    The EEG time point (in seconds) for which to create the RSM.
total_rsm_splits : int
    Number of total RSM splits.
rsm_split : int
    Integer indicating the RSM partition to create. To reduce compute time the
    RSM creation is split into multiple partitions that can run in parallel.
berg_dir : str
    Directory of the Brain Encoding Response Generator.
    https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from berg import BERG
import h5py
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='eeg-things_eeg_2-vit_b_32')
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--channels', default='O,P', type=lambda s: s.split(','))
parser.add_argument('--time', type=float, default=0.1)
parser.add_argument('--total_rsm_splits', type=int, default=5)
parser.add_argument('--rsm_split', type=int, default=1)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Create RSMs <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the in silico EEG responses
# =============================================================================
# Load the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata = berg.get_model_metadata(
    args.encoding_model,
    subject=args.subject
    )

# Get the indices of the selected time point
times = metadata['eeg']['times']
idx_time = np.where(np.round(times, 3) == np.round(args.time, 3))[0][0]

# Load the in silico EEG responses of the selected time point, and average them
# across repeats
data_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'insilico_eeg',
    'insilico_eeg_responses_sub-'+format(args.subject, '02')+'_channels-'+
    '-'.join(args.channels)+'.h5')
eeg = h5py.File(data_dir).get('insilico_eeg_responses')
eeg = np.mean(eeg[:,:,:,idx_time], 1)


# =============================================================================
# Create the in silico EEG response RSM
# =============================================================================
# Establish which RSM partition to create
all_img_cond = np.arange(len(eeg))
img_per_split = int(np.ceil(len(eeg) / args.total_rsm_splits))
idx_start = int(img_per_split * (args.rsm_split-1))
idx_end = int(img_per_split * (args.rsm_split))
if args.rsm_split == args.total_rsm_splits:
    idx_end = len(eeg)
used_img_cond = all_img_cond[idx_start:idx_end]

# Create the RSM
betas_rsm = np.zeros((len(used_img_cond), len(eeg)), dtype=np.float32)
for c1, cond_1 in enumerate(tqdm(used_img_cond)):
    for c2 in range(cond_1):
        betas_rsm[c1,c2] = pearsonr(eeg[cond_1], eeg[c2])[0]


# =============================================================================
# Save the RSMs
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc', 'rsms')
os.makedirs(save_dir, exist_ok=True)

file_name = 'rsm_sub-' + format(args.subject, '02') + '_time-' + \
    str(args.time) + '_split-' + format(args.rsm_split, '02') + '.npy'

np.save(os.path.join(save_dir, file_name), betas_rsm)