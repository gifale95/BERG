"""Extract and save the controlling images from the h5py files.

Parameters
----------
roi: str
    Used ROI.
time_window_1_start: float
    The starting point, in seconds, of first time window of interest.
time_window_1_end: float
    The ending point, in seconds, of first time window of interest.
time_window_2_start: float
    The starting point, in seconds, of second time window of interest.
time_window_2_end: float
    The ending point, in seconds, of second time window of interest.
berg_dir : str
    Directory of the BERG.
"""

import argparse
import os
import h5py
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--time_window_1_start', default=0.06, type=float)
parser.add_argument('--time_window_1_end', default=0.1, type=float)
parser.add_argument('--time_window_2_start', default=0.1, type=float)
parser.add_argument('--time_window_2_end', default=0.2, type=float)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Extract controlling images <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Extract and save the beaseline images
# =============================================================================
# Data directory
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion', 'rnc',
    'images', (f'time_window_1-{args.time_window_1_start}_'
    f'{args.time_window_1_end}__time_window_2-{args.time_window_2_start}_'
    f'{args.time_window_2_end}'), f'roi-{args.roi}')

# Loop across time windows
time_windows = ['time_window_1', 'time_window_2']
for tw in tqdm(time_windows):

    # Load the image h5py file
    h5_dir = os.path.join(data_dir, f'baseline_images_{tw}.h5')
    images = h5py.File(h5_dir, 'r')['images'][:]

    # Save the controlling images as .png files
    for i in range(len(images)):
        img = Image.fromarray(images[i])
        file_name = f'{args.roi}_baseline_{tw}_img-{i+1:03}.png'
        img.save(os.path.join(data_dir, file_name))


# =============================================================================
# Extract and save the controlling images
# =============================================================================
# Loop across neural control types
control_types = ['high_1_high_2', 'low_1_low_2', 'high_1_low_2',
    'low_1_high_2']
for ct in tqdm(control_types):

    # Load the image h5py file
    h5_dir = os.path.join(data_dir, f'controlling_images_{ct}.h5')
    images = h5py.File(h5_dir, 'r')['images'][:]

    # Save the controlling images as .png files
    for i in range(len(images)):
        img = Image.fromarray(images[i])
        file_name = f'{args.roi}_{ct}_img-{i+1:03}.png'
        img.save(os.path.join(data_dir, file_name))