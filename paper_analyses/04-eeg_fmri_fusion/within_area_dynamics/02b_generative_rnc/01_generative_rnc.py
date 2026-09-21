"""Generative univariate RNC iteratively generates images following two serial
objectives. Throughout the genetic optimization generations, the generated
images first drive or suppress the univariate t-fMRI responses of two time
windows up to a threshold. Once this threshold is reached, the image complexity
– as measured by their PNG compression file size – starts to monotonically
decrease, while keeping the t-fMRI responses over the threshold, thus promoting
the generation of images containing only the visual properties necessary to
align or disentangle the two areas.

Parameters
----------
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
eeg_subjects : list
    List containing the subject identifiers for the THINGS EEG2 subjects. Valid
    subject identifiers are integers from 1 to 10.
cv : int
    If '1' univariate RNC leaves the data of one subject out for
    cross-validation, if '0' univariate RNC uses the data of all subjects.
cv_subject : int
    If 'cv==0' the left-out subject during cross-validation, out of all 8 NSD
    subjects.
roi : str
    Used ROI.
hemispheres : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
time_window_pair: str
    A string specifying the two time windows of interest.
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
control_type : str
    If 'high_1_high_2', generate images that drive both time windows. If
    'high_1_low_2', generate images that drive the first time window while
    suppressing the second time window. If 'low_1_high_2', generate images that
    suppress the first time window while driving the second time window. If
    'low_1_low_2', generate images that suppress both time windows.
evolution : int
    Genetic optimization evolution. At each evolution the genetic optimization
    starts from a different random seed, resulting in different controlling
    images.
generations : int
    Number of gemetic optimization generations.
n_image_codes : int
    Number of image code, indicading how many images are generated and evaluated
    at each generation.
image_generator_name : str
    Name of the used image generator. Available options are 'DeePSiM' (a GAN).
    or 'cd_imagenet64_l2' (a class-conditioned diffusion model trained on the
    1000 ILSVRC-2012 classes).
image_generator_class : int
    Integer between 0 and 999 indicating the ILSVRC-2012 class the generated
    image belongs to (if image_generator_name=='cd_imagenet64_l2').
img_complexity_measure : str
    How to compute image complexity. Possbile methods are ['png', 'jpg'].
frac_kept_image_codes : float
    Fraction [0 1] of best image codes that passed onto the next generation
    without being recombined or mutated.
heritability : float
    Fraction [0 1] determining how much one image code parent (of 2) contributes
    to each image code child.
mutation_prob : float
    Probability [0 1] of each new image code genes to be mutated.
imageset : str
    Imageset from which the univariate RNC baseline scores have been computed.
    Possible choices are 'imagenet_val', 'imagenet_train', 'coco'.
project_dir : str
    Directory of the project folder.
berg_dir : str
    Directory of the BERG.
    https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
import random
from tqdm import tqdm
import torch
from PIL import Image
import h5py
from copy import copy
from berg import BERG

from utils import load_encoding_models
from utils import load_image_generator
from utils import generate_tfmri
from utils import score_select
from utils import optimize_image_codes

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--cv', type=int, default=0)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--time_window_pair', default='0.06-0.10__0.20-0.25', type=str)
parser.add_argument('--ncsnr_threshold', type=float, default=0.2)
parser.add_argument('--control_type', type=str, default='high_1_low_2')
parser.add_argument('--generations', type=int, default=500)
parser.add_argument('--evolution', type=int, default=1)
parser.add_argument('--n_image_codes', type=int, default=1000)
parser.add_argument('--image_generator_name', type=str, default='DeePSiM')
parser.add_argument('--image_generator_class', type=int, default=0)
parser.add_argument('--img_complexity_measure', type=str, default='png')
parser.add_argument('--frac_kept_image_codes', type=float, default=.25)
parser.add_argument('--heritability', type=float, default=.25)
parser.add_argument('--mutation_prob', type=float, default=.25)
parser.add_argument('--imageset', type=str, default='imagenet_val')
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generative univariate RNC <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Random seed and device
# =============================================================================
# Set random seed for reproducible results
seed = args.evolution
np.random.seed(seed)
random.seed(seed)
random_generator = np.random.RandomState(seed=seed)
torch.manual_seed(seed)

# Compute device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Break down the time windows
# =============================================================================
# Get the time window start and end times
time_window_1_start, time_window_1_end = map(
    float, args.time_window_pair.split('__')[0].split('-'))
time_window_2_start, time_window_2_end = map(
    float, args.time_window_pair.split('__')[1].split('-'))

# Get the EEG time points
berg = BERG(berg_dir=args.berg_dir)
metadata_eeg = berg.get_model_metadata(
    'eeg-things_eeg_2-vit_b_32',
    subject=1
)
times = np.round(metadata_eeg['eeg']['times'], 3)

# Get the time window indices
t_min_1 = np.where(times == time_window_1_start)[0][0]
t_max_1 = np.where(times == time_window_1_end)[0][0]
t_min_2 = np.where(times == time_window_2_start)[0][0]
t_max_2 = np.where(times == time_window_2_end)[0][0]
n_times = len(times)


# =============================================================================
# Get the fMRI ROI indices
# =============================================================================
# Loop across subjects
idx_v = {}
for fsub in args.fmri_subjects:

    # Load the fMRI metadata
    berg = BERG(berg_dir=args.berg_dir)
    metadata_fmri = berg.get_model_metadata(
        'fmri-nsd_fsaverage-huze',
        subject=fsub
        )

    # Loop across hemisphers
    for h, hemi in enumerate(args.hemispheres):

        # Only select vertices falling within the NSD visual streams
        n_vertices = 163842
        idx_streams = np.zeros(n_vertices, dtype=bool)
        streams = ['early', 'midventral', 'midlateral', 'midparietal',
            'ventral', 'lateral', 'parietal']
        for stream in streams:
            idx_streams[metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][stream]] = 1
        idx_streams = np.where(idx_streams)[0]

        # Only select stream vertices with NCSNR above threshold
        ncsnr = metadata_fmri['fmri'][f'{hemi}_ncsnr']
        idx_ncsnr = np.where(ncsnr[idx_streams] >= args.ncsnr_threshold)[0]

        # Only select stream vertices of the chosen ROI
        if args.roi in ['V1', 'V2', 'V3']:
            idx_r = np.append(
                metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{args.roi}v'],
                metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{args.roi}d'])
            idx_r.sort()
        elif args.roi in ['FFA', 'VWFA', 'FBA']:
            idx_r = np.append(
                metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{args.roi}-1'],
                metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{args.roi}-2'])
            idx_r.sort()
        else:
            idx_r = metadata_fmri['fmri'][f'{hemi}_fsaverage_rois'][f'{args.roi}']
            idx_r.sort()
        idx_roi = np.zeros(n_vertices, dtype=bool)
        idx_roi[idx_r] = 1
        idx_roi = idx_roi[idx_streams]
        idx_roi = np.where(idx_roi)[0]

        # Get the indices of ROI vertices with NCSNR above threshold
        idx_v[(fsub,hemi)] = np.intersect1d(idx_roi, idx_ncsnr)


# =============================================================================
# Load the baseline scores and t-fMRI responses for all images, and compute the
# baseline margin
# =============================================================================
# Load the univariate RNC baseline scores, and average them across images
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'baseline', f'cv-{args.cv}',
    args.time_window_pair, f'imageset-{args.imageset}')
if args.cv == 0:
    file_name = f'baseline_roi-{args.roi}.npy'
    data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
    baseline_tw_1 = np.mean(data['baseline_resp']['time_window_1'])
    baseline_tw_2 = np.mean(data['baseline_resp']['time_window_2'])
elif args.cv == 1:
    file_name = f'baseline_cv_subject-{args.cv_subject}_roi-{args.roi}.npy'
    data = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
    baseline_tw_1 = np.mean(data['baseline_resp_train']['time_window_1'])
    baseline_tw_2 = np.mean(data['baseline_resp_train']['time_window_2'])

# Load the t-fMRI responses of all subjects
tfmri = []
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'tfmri_responses')
for sub in args.fmri_subjects:
    file_name = f'tfmri_sub-{sub:02d}_roi-{args.roi}_imageset-{args.imageset}.h5'
    tfmri.append(h5py.File(os.path.join(data_dir, file_name), 'r')['tfmri'])
tfmri = np.array(tfmri)
# If cross-validating, remove the CV (test) subject, and average over the
# remaining (train) subjects. The fMRI responses for the train subjects are
# used to select the controlling images, and the controlling images will then
# be validated on the fMRI responses for the test subjects. If not
# cross-validating, average over all subjects.
if args.cv == 0:
    tfmri_mean = np.mean(tfmri, 0)
elif args.cv == 1:
    tfmri_mean = np.delete(tfmri, args.cv_subject-1, 0)
    tfmri_mean = np.mean(tfmri_mean, 0)
del tfmri
# Average the t-fMRI responses within the two time windows of interest
tfmri_1 = np.mean(tfmri_mean[:,t_min_1:t_max_1], 1)
tfmri_2 = np.mean(tfmri_mean[:,t_min_2:t_max_2], 1)

# Univariate response score margin used to constrain the selection of the
# control images. The margin is defined as the standard deviation of the
# t-fMRI responses across all images for each time window. The margin is used
# to ignore images that have t-fMRI responses that are too close to the
# baseline scores, as these images may not be informative for aligning or
# disentangling the two time windows. 
margin_tw_1 = np.std(tfmri_1)
margin_tw_2 = np.std(tfmri_2)


# =============================================================================
# Load the EEG and EEG-to-fMRI encoding models of all subjects
# =============================================================================
# Load the models for the first time window
model_eeg_tw_1, model_tfmri_tw_1 = load_encoding_models(args, idx_v, t_min_1,
    t_max_1, n_times, device)

# Load the models for the second time window
model_eeg_tw_2, model_tfmri_tw_2 = load_encoding_models(args, idx_v, t_min_2,
    t_max_2, n_times, device)


# =============================================================================
# Import the image generator
# =============================================================================
image_generator = load_image_generator(args, device)

if args.image_generator_name == 'DeePSiM':
    model_save_dir = 'image_generator-' + args.image_generator_name
elif args.image_generator_name == 'cd_imagenet64_l2':
    model_save_dir = 'image_generator-' + args.image_generator_name + '/' + \
        'image_generator_class-' + format(args.image_generator_class+1, '04')
    random_generator_diffusion = torch.Generator(device=device)


# =============================================================================
# Initialize the image codes
# =============================================================================
# Get the image codes dimensionality
if args.image_generator_name == 'DeePSiM':
    image_code_size = (args.n_image_codes, image_generator.fc7.in_features)
elif args.image_generator_name == 'cd_imagenet64_l2':
    image_code_size = (args.n_image_codes, 3, 64, 64)

# Randomly initialize image codes from a normal distributions
image_codes_new = random_generator.normal(loc=0, scale=1, size=image_code_size)

# Number of kept image codes at each generation
n_kept = int(len(image_codes_new) * args.frac_kept_image_codes)
images_kept = np.empty(0)
image_codes_kept = np.empty(0)
tfmri_tw_1_kept = np.empty(0)
tfmri_tw_2_kept = np.empty(0)


# =============================================================================
# Results variables
# =============================================================================
# Neural control scores
best_neural_control_scores_train = np.zeros(args.generations, dtype=np.float32)
best_neural_control_scores_test = np.zeros(args.generations, dtype=np.float32)
# Penalty scores
best_baseline_penalty_train = np.zeros(args.generations, dtype=np.float32)
best_baseline_penalty_test = np.zeros(args.generations, dtype=np.float32)
# Image complexity scores
best_images_complexity = np.zeros(args.generations, dtype=np.float32)
# Total scores
best_scores_train = np.zeros(args.generations, dtype=np.float32)
best_scores_test = np.zeros(args.generations, dtype=np.float32)
# Image codes
total_image_code_size = (args.generations,) + image_code_size
best_image_codes = np.zeros(total_image_code_size, dtype=np.float32)
# In silico fMRI responses
best_tfmri_tw_1 = np.zeros((args.generations, len(args.fmri_subjects)),
    dtype=np.float32)
best_tfmri_tw_2 = np.zeros((args.generations, len(args.fmri_subjects)),
    dtype=np.float32)


# =============================================================================
# Generate images from the image codes
# =============================================================================
# Generation loop
for g in tqdm(range(args.generations), leave=False):

    img_codes = torch.FloatTensor(copy(image_codes_new))
#	img_codes.to(device)

    # Generate the images using a GAN (DeePSiM)
    if args.image_generator_name == 'DeePSiM':
        # Generate the images
        images_new = image_generator.forward(img_codes).detach().numpy()
        # Clip and scale the images synthesized by the image generator: clamp
        # the output image pixel values to the range [0 255]
        images_new = np.clip(images_new, a_min=0, a_max=255)
        # ======
        # Version 2 (as in Ponce et al., 2019):
        # """To synthesize an image from an input image code, we forward
        # propagated the code through the generative network, clamped the
        # output image pixel values to the valid range between 0 and 1, and
        # visualized them as an 8-bit color image."""
        #images_new = np.clip(images_new, a_min=0, a_max=1) * 255
        # ======
        # Version 3:
        # Clamp the output image pixel values to the range [-255 255],
        # normalize them in the range [0 1], and scale them to the range
        # [0 255]
        #images_new = np.clip(images_new, a_min=-255, a_max=255)
        #images_new = (images_new - np.min(images_new.flatten())) / \
        #	(np.max(images_new.flatten()) - np.min(images_new.flatten())) * 255

    # Generate the images using a diffusion model (cd_imagenet64_l2)
    elif args.image_generator_name == 'cd_imagenet64_l2':
        # Generate the image codes into two batches (for GPU RAM)
        batch_n = 2
        batch_size = int(np.ceil(len(img_codes) / batch_n))
        class_labels = [args.image_generator_class] * batch_size
        for b in range(batch_n):
            idx_start = batch_size * b
            idx_end = idx_start + batch_size
            # Set a constant random seed to enforce a deterministic image
            # generation
            #torch.manual_seed(seed) # Used to determine the image class
            random_generator_diffusion.manual_seed(seed) # Used to determine the image style
            # Generate the images
            with torch.inference_mode():
                images_new_batch = image_generator(
                    batch_size=batch_size,
                    class_labels=class_labels,
                    num_inference_steps=40,
                    generator=random_generator_diffusion,
                    latents=img_codes[idx_start:idx_end],
                    output_type='np'
                    ).images
                if b == 0:
                    images_new = images_new_batch
                else:
                    images_new = np.append(images_new, images_new_batch, 0)
                del images_new_batch
        # Reshape to (Batch size x 3 RGB Channels x Width x Height)
        images_new = np.transpose(images_new, (0, 3, 1, 2))
        # Scale to the range [0, 255]
        images_new *= 255

    # Convert the images to uint8
    images_new = images_new.astype(np.uint8)
    del img_codes


# =============================================================================
# Generate t-fMRI responses for the synthesized images
# =============================================================================
    # Generate the t-fMRI responses for the first time window
    tfmri_tw_1_new = generate_tfmri(args, model_eeg_tw_1,
        model_tfmri_tw_1, copy(images_new), berg)

    # Generate the t-fMRI responses for the second time window
    tfmri_tw_2_new = generate_tfmri(args, model_eeg_tw_2,
        model_tfmri_tw_2, copy(images_new), berg)


# =============================================================================
# Add kept data from the previous generation
# =============================================================================
    if g == 0:
        image_codes = image_codes_new
        images = images_new
        tfmri_tw_1 = tfmri_tw_1_new
        tfmri_tw_2 = tfmri_tw_2_new
    else:
        image_codes = np.append(image_codes_kept, image_codes_new, 0)
        images = np.append(images_kept, images_new, 0)
        tfmri_tw_1 = np.append(tfmri_tw_1_kept, tfmri_tw_1_new, 1)
        tfmri_tw_2 = np.append(tfmri_tw_2_kept, tfmri_tw_2_new, 1)

    del image_codes_kept, image_codes_new, images_kept, images_new, \
        tfmri_tw_1_kept, tfmri_tw_1_new, tfmri_tw_2_kept, tfmri_tw_2_new


# =============================================================================
# Compute the neural control scores, and select the image codes accordingly
# =============================================================================
    # Score the generated images, rank the scores, and then select/store the
    # image codes of the best N images
    scores_train, scores_test, neural_control_scores_train, \
        neural_control_scores_test, baseline_penalty_train, \
        baseline_penalty_test, images_complexity, image_codes, tfmri_tw_1, \
        tfmri_tw_2, images = score_select(args, tfmri_tw_1, tfmri_tw_2,
        image_codes, images, baseline_tw_1, baseline_tw_2, margin_tw_1,
        margin_tw_2)

    # Save the best scores, image codes and fMRI responses of each generation
    best_scores_train[g] = scores_train[0]
    best_scores_test[g] = scores_test[0]
    best_neural_control_scores_train[g] = neural_control_scores_train[0]
    best_neural_control_scores_test[g] = neural_control_scores_test[0]
    best_baseline_penalty_train[g] = baseline_penalty_train[0]
    best_baseline_penalty_test[g] = baseline_penalty_test[0]
    best_images_complexity[g] = images_complexity[0]
    best_image_codes[g] = image_codes[0]
    best_tfmri_tw_1[g] = tfmri_tw_1[:,0]
    best_tfmri_tw_2[g] = tfmri_tw_2[:,0]


# =============================================================================
# Store the results from the kept image codes to reduce computation
# =============================================================================
    image_codes_kept = copy(image_codes[:n_kept])
    images_kept = copy(images[:n_kept])
    tfmri_tw_1_kept = copy(tfmri_tw_1[:,:n_kept])
    tfmri_tw_2_kept = copy(tfmri_tw_2[:,:n_kept])
    del images, tfmri_tw_1, tfmri_tw_2


# =============================================================================
# Optimize the image codes using a genetic algorithm
# =============================================================================
    image_codes_new = optimize_image_codes(args, image_codes,
        copy(scores_train), random_generator)
    del image_codes


# =============================================================================
# Save the best image of every Nth generation
# =============================================================================
    if args.cv == 0 and (g+1) % 1 == 0:

        if args.cv == 0:
            save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
                'within_area_dynamics', 'generative_univariate_rnc',
                'controlling_images', 'cv-'+f'{args.cv}',
                args.time_window_pair, 'control_condition-'+args.control_type,
                model_save_dir, 'evolution-'+f'{args.evolution:02d}')
        elif args.cv == 1:
            save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
                'within_area_dynamics', 'generative_univariate_rnc',
                'controlling_images', 'cv-'+f'{args.cv}',
                args.time_window_pair, 'control_condition-'+args.control_type,
                'cv_subject-'+f'{args.cv_subject:02d}', model_save_dir,
                'evolution-'+f'{args.evolution:02d}')

        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)

        for i in range(args.n_image_codes):
            img = Image.fromarray(np.swapaxes(np.swapaxes(
                images_kept[0], 0, 1), 1, 2))
            file_name = 'gan_img_' + args.control_type + \
                '_generation-' + f'{g+1:05d}' + '_null_penalty-' + \
                str(best_baseline_penalty_train[g]) + '_complexity-' + \
                f'{best_images_complexity[g]:08f}' + '.png'
            img.save(os.path.join(save_dir, file_name))


# =============================================================================
# Save the optimization scores
# =============================================================================
data_dict = {
    'args': args,
    'best_scores_train': best_scores_train,
    'best_scores_test': best_scores_test,
    'best_neural_control_scores_train': best_neural_control_scores_train,
    'best_neural_control_scores_test': best_neural_control_scores_test,
    'best_baseline_penalty_train': best_baseline_penalty_train,
    'best_baseline_penalty_test': best_baseline_penalty_test,
    'best_images_complexity': best_images_complexity,
    'best_tfmri_tw_1': best_tfmri_tw_1,
    'best_tfmri_tw_2': best_tfmri_tw_2,
    'baseline_score_train_tw_1': baseline_tw_1,
    'baseline_score_train_tw_2': baseline_tw_2
    }

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'generative_univariate_rnc', 'optimization_scores',
    'cv-'+f'{args.cv}', args.time_window_pair, 'control_condition-'+
    args.control_type, model_save_dir, 'evolution-'+f'{args.evolution:02d}')

if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

if args.cv == 0:
    file_name = 'optimization_scores'
elif args.cv == 1:
    file_name = f'optimization_scores_cv_subject-{args.cv_subject:02d}'

np.save(os.path.join(save_dir, file_name), data_dict)


# =============================================================================
# Save the image codes
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'generative_univariate_rnc', 'image_codes',
    'cv-'+f'{args.cv}', args.time_window_pair, 'control_condition-'+
    args.control_type, model_save_dir, 'evolution-'+f'{args.evolution:02d}')

if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

if args.cv == 0:
    file_name = 'image_codes'
elif args.cv == 1:
    file_name = f'image_codes_cv_subject-{args.cv_subject:02d}'

np.save(os.path.join(save_dir, file_name), np.asarray(best_image_codes))