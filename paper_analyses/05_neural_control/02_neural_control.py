"""Apply neural control to find images that drive or suppress the in silico
monkey electrophysiology responses. The controlling images are then
cross-validated across subjects.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico 
    responses.
subjects : list
    List of subject identifiers for the monkey encoding model. Since the used
    encoding models are trained on the TVSD data, valid subject identifiers
    are "N" and "F".
roi: str
    ROI used. Valid values are "V1", "V4", and "IT".
time_resolved: int
    If 1, apply neural control in a time-reslved fashion (i.e., a separate
    controlling image is found for each time point).
    If 0, control the average neural response across the time window around
    peak activity (V1: 25ms-125ms; V4: 50ms-150ms; IT: 75ms-175ms).
control: str
    Whether to "drive" or "suppress" neural responses.
n_controlling_imgs: int
    Number of retained controlling images.
berg_dir : str
    Directory of the BERG.
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import argparse
import os
import numpy as np
import random
import torchvision
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subjects', default=['N', 'F'], type=list)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--time_resolved', default=1, type=int)
parser.add_argument('--control', default='drive', type=str)
parser.add_argument('--n_controlling_imgs', default=10, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/giffordale95/datasets/image_sets/ILSVRC2012', type=str)
args, unknown = parser.parse_known_args()

print('>>> Neural control <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the in silico neural responses for the ~1.3M ILSVRC-2012 train images
# =============================================================================
# Load the in silico responses
data_dir = os.path.join(args.berg_dir, 'neural_control', 'insilico_responses',
    args.encoding_model)
insilico_data = []
metadata = []
for sub in args.subjects:
    file_name = f'insilico_responses_sub-{sub}_roi-{args.roi}.npy'
    data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
    insilico_data.append(data['responses'])
    metadata.append(data['metadata'])
insilico_data = np.array(insilico_data)

# Optionally average the in silico neural responses across the time window
# around peak activity
if args.time_resolved == 1:
    peaks = {
        'V1': (25, 125),
        'V4': (50, 150),
        'IT': (75, 175)
    }
    times = metadata[0]['utah_array']['times']
    t_min = np.where(times == peaks[args.roi][0])[0] # !!! CHECK IF CORRECT
    t_max = np.where(times == peaks[args.roi][1])[0] # !!! CHECK IF CORRECT
    insilico_data = np.mean(insilico_data[:,:,t_min:t_max], 2)


# =============================================================================
# Neural control
# =============================================================================
# Find the images controlling the neural responses
if args.control == 'drive':
    img_control = np.argsort(insilico_data, axis=1)[:,::-1]
elif args.control == 'suppress':
    img_control = np.argsort(insilico_data, axis=1)
img_control = img_control[:,:args.n_controlling_imgs]

# Cross-validate the controlling images across subjects
control_data = []
cv_control_data = []
for s in range(len(img_control)):
    control_data.append(insilico_data[s,img_control[s]]) # !!! HOW TO SORT ALL DIMENSIONS OF A 2-3D ARRAY?
    s_cv = np.delete((0, 1), s)[0]
    cv_control_data.append(insilico_data[s_cv,img_control[s]]) # !!! HOW TO SORT ALL DIMENSIONS OF A 2-3D ARRAY?


# =============================================================================
# Baseline # !!! GET CODE FROM RNC
# =============================================================================
# Find the baseline images
img_baseline = 

# Cross-validate the baseline images across subjects
baseline_data = 
cv_baseline_data = 


# =============================================================================
# Save the quantitative neural control results
# =============================================================================
results = {
    'img_control': img_control,
    'control_data': control_data,
    'cv_control_data': cv_control_data,
    'img_baseline': img_baseline,
    'baseline_data': baseline_data,
    'cv_baseline_data': cv_baseline_data
}

save_dir = os.path.join(args.berg_dir, 'neural_control',
    'quantitative_results', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = (f'neural_control_roi-{args.roi}'
            f'_time_resolved-{args.time_resolved}_control-{args.control}.npy')

np.save(os.path.join(save_dir, file_name), results)


# =============================================================================
# Save the controlling and baseline images # !!!
# =============================================================================
# Access the ILSVRC-2012 train split
images = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='train')

# Define the image preprocessing # !!! EDIT THIS, AND MAKE SURE THAT PIL IMAGE IS LOADED
transform = trn.Compose([
    trn.Lambda(lambda img: trn.functional.center_crop(img, min(img.size))),
    trn.Resize((224, 224)),
    trn.Lambda(lambda img: np.transpose(img, (2, 0, 1))) # HWC to CHW
])

# Save the controlling and baseline images
for s, sub in enumerate(tqdm(args.subjects)):
    save_dir_control = os.path.join(args.berg_dir, 'neural_control',
        'controlling images', args.encoding_model, f'subject-{sub}',
        f'roi-{args.roi}', f'time_resolved-{args.time_resolved}',
        f'control-{args.control}')
    save_dir_baseline = os.path.join(args.berg_dir, 'neural_control',
        'controlling images', args.encoding_model, f'subject-{sub}',
        f'roi-{args.roi}', f'time_resolved-{args.time_resolved}',
        f'control-{args.control}')
    os.makedirs(save_dir, exist_ok=True)
    if args.time_resolved == 1:
        for t, time in enumerate(times):
            for i in range(args.n_controlling_imgs):
                img_control = # !!!
                img_baseline = # !!!
                if str(time)[0] == '-':
                    file_name = f'-{abs(time):03}ms_img-{i:03}.png'
                else:
                    file_name = f'+{time:03}ms_img-{i:03}.png'
                # !!! Save the controlling image
                # !!! Save the baseline image
    else:
        for i in range(args.n_controlling_imgs):
            img_control = # !!!
            img_baseline = # !!!
            file_name = f'img-{i:03}.png'
            # !!! Save the controlling image
            # !!! Save the baseline image