"""For each pairwise time combination multivariate RNC uses genetic
optimization and representational similarity analysis (RSA) to search for a
batch of images that aligns (i.e., images leading to a high RSA correlation
score) or disentangles (i.e., images leading to a low absolute RSA correlation
score) the in silico multivariate EEG responses for the two time points being
compared, thus highlighting shared and unique representational content,
respectively.

Parameters
----------
cv : int
    If '1' multivariate RNC leaves the data of one subject out for
    cross-validation, if '0' multivariate RNC uses the data of all subjects.
cv_subject : int
    If cv==1, the left-out subject during cross-validation, out of the 10
    THINGS EEG2 subjects.
time_pair : str
    Used pairwise time point combination.
control_condition : str
    Whether to 'align' or 'disentangle' the multivariate EEG responses for the
    two time points being compared.
generations : int
    Number of genetic optimization generations.
n_batches : int
    Initial amount of image batches at each genetic optimization generation.
n_images_per_batch : int
    Amount of images per image batch, that is, of the controlling images.
berg_dir : str
    Directory of the Brain Encoding Response Generator.
    https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
import random
from tqdm import tqdm

from utils import load_rsms
from utils import create_batches
from utils import mutate
from utils import evaluate
from utils import select

parser = argparse.ArgumentParser()
parser.add_argument('--cv', type=int, default=0)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--time_pair', type=str, default='0.1-0.2')
parser.add_argument('--control_condition', type=str, default='disentangle')
parser.add_argument('--generations', type=int, default=2000)
parser.add_argument('--n_batches', type=int, default=200)
parser.add_argument('--n_images_per_batch', type=int, default=50)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Multivariate RNC <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)


# =============================================================================
# Time point names
# =============================================================================
idx = args.time_pair.find('-')
time_1 = args.time_pair[:idx]
time_2 = args.time_pair[idx+1:]


# =============================================================================
# Load the pre-computed in silico fMRI RSMs
# =============================================================================
if args.cv == 0:
    # If not cross-validating, load and use the RSMs averaged across all
    # subjects.
    rsm_1, rsm_2 = load_rsms(args)

elif args.cv == 1:
    # If cross-validating, load and use the RSMs of the N-1 subjects (i.e., the
    # 7 remaining subjects beyond the 'cv_subject').
    rsm_1, rsm_2 = load_rsms(args, args.cv_subject, 'train')


# =============================================================================
# Use genetic optimization to find aligning and disentangling images
# =============================================================================
# Randomly create the first generation of image batches
image_batches = create_batches(args.n_batches, args.n_images_per_batch,
    len(rsm_1))
image_batches_scores = np.zeros((args.generations))
best_generation_image_batches = np.zeros((args.generations,
    args.n_images_per_batch), dtype=int)

for g in tqdm(range(args.generations)):
    # At the beginning of each genetic optimization generation the image
    # batches are augmented following exploitation and exploration.
    # Exploitation involves creating five mutated versions for each of the
    # image batches, where in each version a different amount of batch images
    # is randomly replaced with other images from the ROIs RSMs (while ensuring
    # that no image is repeated within the same batch). Exploration involves
    # creating new random batches.
    # Augment the image batches via mutations (exploitation)
    mutated_image_batches = mutate(image_batches, len(rsm_1))
    image_batches = np.append(image_batches, mutated_image_batches, 0)
    # Augment the image batches with new random batches (exploration)
    new_image_batches = create_batches(len(image_batches),
        args.n_images_per_batch, len(rsm_1))
    image_batches = np.append(image_batches, new_image_batches, 0)
    image_batches.sort(1)

    # Perform RSA between the two ROIs (i.e., correlate the RSMs of the two
    # ROIs) using only the RSM entries corresponding to the images from the
    # used image batches, resulting in one RSA correlation score per batch.
    scores = evaluate(image_batches, rsm_1, rsm_2)

    # To align the two ROIs, keep the N image batches (where N is defined by
    # the 'n_batches' variable) with highest correlation scores (i.e.,
    # containing images most similarly represented by the two ROIs), whereas to
    # disentangle them keep the N image batches with lowest absolute
    # correlation scores (i.e., containing images most differently represented
    # by the two ROIs). These image batches are then passed to the next genetic
    # otpimization generation, where the same steps are repeated.
    image_batches, scores = select(args.control_condition, image_batches,
        scores, args.n_batches)
    # Store the best image batch of each generation, along with its score
    image_batches_scores[g] = scores[0]
    best_generation_image_batches[g] = image_batches[0]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'time_1': time_1,
    'time_2': time_2,
    'image_batches_scores': image_batches_scores,
    'best_generation_image_batches': best_generation_image_batches
    }

save_dir = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc',
    'best_image_batches', 'cv-'+format(args.cv), args.time_pair,
    'control_condition-'+args.control_condition)
os.makedirs(save_dir, exist_ok=True)

if args.cv == 0:
    file_name = 'best_image_batches.npy'
elif args.cv == 1:
    file_name = 'best_image_batches_subject-' + \
        format(args.cv_subject, '02') + '.npy'

np.save(os.path.join(save_dir, file_name), results)