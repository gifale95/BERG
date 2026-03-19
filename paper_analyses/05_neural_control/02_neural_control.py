"""Apply neural control to find images that drive or suppress the in silico
monkey electrophysiology responses. The controlling images are then
cross-validated across subjects.

Two types of neural control are performed. One finds a controlling image for
each time point, whereas the other find a single controlling image across based
on the neural activity averaged across the time window around peak activity.
For both types of neural control, the controlling images are then
cross-validated across subjects in a tim-resolved fashion.

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
control: str
    Whether to "drive" or "suppress" neural responses.
n_images: int
    Number of retained controlling or baseline images.
n_iter : int
    Amount of iterations to generate the bootstrap and null distributions.
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
from sklearn.utils import resample
from copy import copy
from torchvision import transforms as trn
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='utah_array-tvsd-vit_b_32')
parser.add_argument('--subjects', default=['N', 'F'], type=list)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--control', default='drive', type=str)
parser.add_argument('--n_images', default=10, type=int)
parser.add_argument('--n_iter', type=int, default=100000)
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

# Average the in silico neural responses across the time window around peak
# activity
times = metadata[0]['utah_array']['times']
peaks = {
    'V1': (25, 125),
    'V4': (50, 150),
    'IT': (75, 175)
}
t_min = np.where(times == peaks[args.roi][0])[0][0]
t_max = np.where(times == peaks[args.roi][1])[0][0]
insilico_data_avg = np.mean(insilico_data[:,:,t_min:t_max], 2)


# =============================================================================
# Neural control
# =============================================================================
# Find the images controlling the neural responses
if args.control == 'drive':
    img_control = np.argsort(insilico_data, axis=1)[:,::-1]
    img_control_avg = np.argsort(insilico_data_avg, axis=1)[:,::-1]
elif args.control == 'suppress':
    img_control = np.argsort(insilico_data, axis=1)
    img_control_avg = np.argsort(insilico_data_avg, axis=1)
img_control = img_control[:,:args.n_images]
img_control_avg = img_control_avg[:,:args.n_images]

# Cross-validate the controlling images across subjects
control_data = []
cv_control_data = []
control_data_avg = []
cv_control_data_avg = []
for s in range(len(args.subjects)):
    s_cv = np.delete((0, 1), s)[0]
    control_data.append(np.take_along_axis(insilico_data[s], img_control[s],
        axis=0))
    cv_control_data.append(np.take_along_axis(insilico_data[s_cv],
        img_control[s], axis=0))
    control_data_avg.append(insilico_data[s,img_control_avg[s]])
    cv_control_data_avg.append(insilico_data[s_cv,img_control_avg[s]])
control_data = np.array(control_data)
cv_control_data = np.array(cv_control_data)
control_data_avg = np.array(control_data_avg)
cv_control_data_avg = np.array(cv_control_data_avg)


# =============================================================================
# Baseline
# =============================================================================
# Get the in silico neural responses for a randomly selected batch of N images
# (out of all images in the image set), and then average these univariate
# responses across images. This will result in one score indicating the mean in
# silico univariate fMRI response for that image batch.
# Repeating this step 1 million times will create the null  distribution, from
# which the N images from the batch with score closest to the distribution's
# mean are selected. The in silico neural response score averaged across these
# N images provides the neural control baseline.

# Create the baseline null distribution
images = insilico_data.shape[1]
null_distribution = []
null_distribution_images = []
for i in tqdm(range(args.n_iter), leave=False):
    sample = resample(np.arange(images), replace=False,
        n_samples=args.n_images)
    sample.sort()
    null_distribution_images.append(copy(sample))
    null_distribution.append(np.mean(
        insilico_data[:,sample], 1).astype(np.float32))
null_distribution = np.array(null_distribution)
null_distribution_images = np.array(null_distribution_images)

# Compute the confidence intervals of the baseline null distribution
ci_low_null_distribution = np.percentile(null_distribution, 2.5, 0)
ci_high_null_distribution = np.percentile(null_distribution, 97.5, 0)

# Select the baseline images from the null distribution (these are the images
# closest to the null distribution mean)
null_distribution_mean = np.mean(null_distribution, 0)
idx_best = np.argsort(abs(null_distribution - null_distribution_mean), 0)[0]
idx_best_avg = np.argsort(abs(np.mean(
    null_distribution[:,:,t_min:t_max], 2) - \
    np.mean(null_distribution_mean[:,t_min:t_max], 1)), 0)[0]
img_baseline = null_distribution_images[idx_best]
img_baseline = np.transpose(img_baseline, (0, 2, 1))
img_baseline_avg = null_distribution_images[idx_best_avg]

# Cross-validate the baseline images across subjects
baseline_data = []
cv_baseline_data = []
baseline_data_avg = []
cv_baseline_data_avg = []
for s in range(len(args.subjects)):
    s_cv = np.delete((0, 1), s)[0]
    baseline_data.append(np.take_along_axis(insilico_data[s], img_baseline[s],
        axis=0))
    cv_baseline_data.append(np.take_along_axis(insilico_data[s_cv],
        img_baseline[s], axis=0))
    baseline_data_avg.append(insilico_data[s,img_baseline_avg[s]])
    cv_baseline_data_avg.append(insilico_data[s_cv,img_baseline_avg[s]])
baseline_data = np.array(baseline_data)
cv_baseline_data = np.array(cv_baseline_data)
baseline_data_avg = np.array(baseline_data_avg)
cv_baseline_data_avg = np.array(cv_baseline_data_avg)


# =============================================================================
# Compute the confidence intervals
# =============================================================================
dist = np.zeros((args.n_iter, len(args.subjects), len(times)))
dist_avg = np.zeros((args.n_iter, len(args.subjects), len(times)))
dist_baseline = np.zeros((args.n_iter, len(args.subjects), len(times)))
dist_baseline_avg = np.zeros((args.n_iter, len(args.subjects), len(times)))
dist_cv = np.zeros((args.n_iter, len(args.subjects), len(times)))
dist_cv_avg = np.zeros((args.n_iter, len(args.subjects), len(times)))
dist_cv_baseline = np.zeros((args.n_iter, len(args.subjects), len(times)))
dist_cv_baseline_avg = np.zeros((args.n_iter, len(args.subjects), len(times)))

for i in tqdm(range(args.n_iter), leave=False):
    idx = resample(np.arange(args.n_images))
    dist[i] = np.mean(control_data[:,idx], axis=1)
    dist_avg[i] = np.mean(control_data_avg[:,idx], axis=1)
    dist_baseline[i] = np.mean(baseline_data[:,idx], axis=1)
    dist_baseline_avg[i] = np.mean(baseline_data_avg[:,idx], axis=1)
    dist_cv[i] = np.mean(cv_control_data[:,idx], axis=1)
    dist_cv_avg[i] = np.mean(cv_control_data_avg[:,idx], axis=1)
    dist_cv_baseline[i] = np.mean(cv_baseline_data[:,idx], axis=1)
    dist_cv_baseline_avg[i] = np.mean(cv_baseline_data_avg[:,idx], axis=1)

ci_low_control_data = np.percentile(dist, 2.5, axis=0)
ci_high_control_data = np.percentile(dist, 97.5, axis=0)
ci_low_control_data_avg = np.percentile(dist_avg, 2.5, axis=0)
ci_high_control_data_avg = np.percentile(dist_avg, 97.5, axis=0)
ci_low_baseline_data = np.percentile(dist_baseline, 2.5, axis=0)
ci_high_baseline_data = np.percentile(dist_baseline, 97.5, axis=0)
ci_low_baseline_data_avg = np.percentile(dist_baseline_avg, 2.5, axis=0)
ci_high_baseline_data_avg = np.percentile(dist_baseline_avg, 97.5, axis=0)
ci_low_cv_control_data = np.percentile(dist_cv, 2.5, axis=0)
ci_high_cv_control_data = np.percentile(dist_cv, 97.5, axis=0)
ci_low_cv_control_data_avg = np.percentile(dist_cv_avg, 2.5, axis=0)
ci_high_cv_control_data_avg = np.percentile(dist_cv_avg, 97.5, axis=0)
ci_low_cv_baseline_data = np.percentile(dist_cv_baseline, 2.5, axis=0)
ci_high_cv_baseline_data = np.percentile(dist_cv_baseline, 97.5, axis=0)
ci_low_cv_baseline_data_avg = np.percentile(dist_cv_baseline_avg, 2.5, axis=0)
ci_high_cv_baseline_data_avg = np.percentile(dist_cv_baseline_avg, 97.5, axis=0)


# =============================================================================
# Compute the significance of the CV neural control scores
# =============================================================================
# Empty p-value lists
p_val = []
p_val_avg = []
p_val_bh = []
p_val_avg_bh = []
p_val_bonf = []
p_val_avg_bonf = []

# Loop across subjects
for s in range(len(args.subjects)):

    # Compute the within-subject p-values
    s_cv = np.delete((0, 1), s)[0]
    if args.control == 'drive':
        idx = np.sum(
            null_distribution[:,s] > np.mean(cv_control_data[s_cv], 0), 0)
        idx_avg = np.sum(
            null_distribution[:,s] > np.mean(cv_control_data_avg[s_cv], 0), 0)
    elif args.control == 'suppress':
        idx = np.sum(
            null_distribution[:,s] < np.mean(cv_control_data[s_cv], 0), 0)
        idx_avg = np.sum(
            null_distribution[:,s] < np.mean(cv_control_data_avg[s_cv], 0), 0)
    p_val_sub = (idx + 1) / (args.n_iter + 1) # Add 1 to avoid p-values of 0
    p_val_avg_sub = (idx_avg + 1) / (args.n_iter + 1)
    p_val.append(p_val_sub)
    p_val_avg.append(p_val_avg_sub)

    # Correct for multiple comparisons
    p_val_bh.append(multipletests(p_val_sub, 0.05, 'fdr_bh')[1])
    p_val_avg_bh.append(multipletests(p_val_avg_sub, 0.05, 'fdr_bh')[1])
    p_val_bonf.append(multipletests(p_val_sub, 0.05, 'bonferroni')[1])
    p_val_avg_bonf.append(multipletests(p_val_avg_sub, 0.05, 'bonferroni')[1])

# Format to numpy arrays
p_val = np.array(p_val)
p_val_avg = np.array(p_val_avg)
p_val_bh = np.array(p_val_bh)
p_val_avg_bh = np.array(p_val_avg_bh)
p_val_bonf = np.array(p_val_bonf)
p_val_avg_bonf = np.array(p_val_avg_bonf)


# =============================================================================
# Save the quantitative neural control results
# =============================================================================
results = {
    'img_control': img_control,
    'control_data': control_data,
    'ci_low_control_data': ci_low_control_data,
    'ci_high_control_data': ci_high_control_data,
    'cv_control_data': cv_control_data,
    'ci_low_cv_control_data': ci_low_cv_control_data,
    'ci_high_cv_control_data': ci_high_cv_control_data,
    'p_val': p_val,
    'p_val_bh': p_val_bh,
    'p_val_bonf': p_val_bonf,

    'img_control_avg': img_control_avg,
    'control_data_avg': control_data_avg,
    'ci_low_control_data_avg': ci_low_control_data_avg,
    'ci_high_control_data_avg': ci_high_control_data_avg,
    'cv_control_data_avg': cv_control_data_avg,
    'ci_low_cv_control_data_avg': ci_low_cv_control_data_avg,
    'ci_high_cv_control_data_avg': ci_high_cv_control_data_avg,
    'p_val_avg': p_val_avg,
    'p_val_avg_bh': p_val_avg_bh,
    'p_val_avg_bonf': p_val_avg_bonf,

    'ci_low_null_distribution': ci_low_null_distribution,
    'ci_high_null_distribution': ci_high_null_distribution,
    'img_baseline': img_baseline,
    'baseline_data': baseline_data,
    'ci_low_baseline_data': ci_low_baseline_data,
    'ci_high_baseline_data': ci_high_baseline_data,
    'cv_baseline_data': cv_baseline_data,
    'ci_low_cv_baseline_data': ci_low_cv_baseline_data,
    'ci_high_cv_baseline_data': ci_high_cv_baseline_data,
    'img_baseline_avg': img_baseline_avg,
    'baseline_data_avg': baseline_data_avg,
    'ci_low_baseline_data_avg': ci_low_baseline_data_avg,
    'ci_high_baseline_data_avg': ci_high_baseline_data_avg,
    'cv_baseline_data_avg': cv_baseline_data_avg,
    'ci_low_cv_baseline_data_avg': ci_low_cv_baseline_data_avg,
    'ci_high_cv_baseline_data_avg': ci_high_cv_baseline_data_avg,
}

save_dir = os.path.join(args.berg_dir, 'neural_control',
    'quantitative_results', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = f'roi-{args.roi}_control-{args.control}.npy'

np.save(os.path.join(save_dir, file_name), results)


# =============================================================================
# Save the controlling and baseline images
# =============================================================================
# Access the ILSVRC-2012 train split
images = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='train')

# Loop across subjects
for s, sub in enumerate(tqdm(args.subjects)):

    # Save the controlling and baseline images for the time-resolved neural
    # control
    save_dir_control = os.path.join(args.berg_dir, 'neural_control',
        'controlling images', args.encoding_model, f'subject-{sub}',
        f'roi-{args.roi}', f'control-{args.control}', 'time-resolved')
    os.makedirs(save_dir_control, exist_ok=True)
    save_dir_baseline = os.path.join(args.berg_dir, 'neural_control',
        'baseline images', args.encoding_model, f'subject-{sub}',
        f'roi-{args.roi}', f'control-{args.control}', 'time-resolved')
    os.makedirs(save_dir_baseline, exist_ok=True)
    for t, time in enumerate(times):
        if t % 10 == 0: # Only same images for every 10ms time point
            for i in range(args.n_images):
                img_c, _ = images.__getitem__(img_control[s,i,t])
                min_size = min(img_c.size)
                transform = trn.Compose([
                    trn.CenterCrop(min_size),
                    trn.Resize((425,425))
                    ])
                img_c = transform(img_c)
                img_b, _ = images.__getitem__(img_baseline[s,i,t])
                min_size = min(img_b.size)
                transform = trn.Compose([
                    trn.CenterCrop(min_size),
                    trn.Resize((425,425))
                    ])
                img_b = transform(img_b)
                if str(time)[0] == '-':
                    file_name_control = \
                        f'{args.control}_-{abs(time):03}ms_img-{i:03}.png'
                    file_name_baseline = \
                        f'baseline_-{abs(time):03}ms_img-{i:03}.png'
                else:
                    file_name_control = \
                        f'{args.control}_+{time:03}ms_img-{i:03}.png'
                    file_name_baseline = \
                        f'baseline_+{time:03}ms_img-{i:03}.png'
                img_c.save(os.path.join(save_dir_control, file_name_control))
                img_b.save(os.path.join(save_dir_baseline, file_name_baseline))

    # Save the controlling and baseline images for the time-averaged neural
    # control
    save_dir_control = os.path.join(args.berg_dir, 'neural_control',
        'controlling images', args.encoding_model, f'subject-{sub}',
        f'roi-{args.roi}', f'control-{args.control}', 'time-averaged')
    os.makedirs(save_dir_control, exist_ok=True)
    save_dir_baseline = os.path.join(args.berg_dir, 'neural_control',
        'baseline images', args.encoding_model, f'subject-{sub}',
        f'roi-{args.roi}', f'control-{args.control}', 'time-averaged')
    os.makedirs(save_dir_baseline, exist_ok=True)
    for i in range(args.n_images):
        img_c, _ = images.__getitem__(img_control_avg[s,i])
        min_size = min(img_c.size)
        transform = trn.Compose([
            trn.CenterCrop(min_size),
            trn.Resize((425,425))
            ])
        img_c = transform(img_c)
        img_b, _ = images.__getitem__(img_baseline_avg[s,i])
        min_size = min(img_b.size)
        transform = trn.Compose([
            trn.CenterCrop(min_size),
            trn.Resize((425,425))
            ])
        img_b = transform(img_b)
        file_name_control = f'{args.control}_-img-{i:03}.png'
        file_name_baseline = f'baseline_img-{i:03}.png'
        img_c.save(os.path.join(save_dir_control, file_name_control))
        img_b.save(os.path.join(save_dir_baseline, file_name_baseline))