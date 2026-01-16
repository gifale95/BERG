"""Create the MEG RDMs through pairwise decoding.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico MEG
    responses.
subject : int
    The subject identifier for the MEG encoding models.
sensors : string
    String containing the MEG sensor type(s) retained for the analyses,
    separated by a comma. Possible values are: 'O' (occipital), 'P'
    (posterior), 'T' (temporal), 'C' (central), 'F' (frontal).
tmax : float
    Maximum epoch time point for the MEG analyses.
berg_dir : str
    Directory of the BERG.
things_dir : str
    Directory of the THINGS database.
    https://osf.io/jum2f/

"""

import argparse
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from berg import BERG
from sklearn.svm import SVC

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='meg-things_meg_1-vit_b_32')
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--sensors', default='O,P', type=lambda s: s.split(','))
parser.add_argument('--tmax', default=0.6, type=float)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--things_dir', default='/scratch/giffordale95/datasets/image_sets/things_database', type=str)
args, unknown = parser.parse_known_args()

print('>>> EEG RDMs <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the stimulus images
# =============================================================================
# Load the test images metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_meg = berg.get_model_metadata(
    args.encoding_model,
    subject=args.subject
)
test_image_nr = metadata_meg['encoding_model']['test_image_nr']
test_img_files = metadata_meg['encoding_model']['test_img_files']
unique_test_img_nr = np.unique(test_image_nr)

# Loop across test images
images = []
for img_nr in tqdm(unique_test_img_nr):

    # Get the image directory
    img_file = test_img_files[np.where(test_image_nr == img_nr)[0][0]]
    img_path = os.path.join(args.things_dir, 'image-database_things', img_file)

    # Load and transform the image
    img = Image.open(img_path)
    img = img.resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
    img = np.array(img).transpose(2, 0, 1)  # Convert to (C, H, W)
    images.append(img)

# Format the images to a numpy array
images = np.array(images)


# =============================================================================
# MEG time point and sensor selection
# =============================================================================
# Time point selection
times = metadata_meg['meg']['times']
timepoints = np.zeros(len(times), dtype=int)
timepoints[times <= args.tmax] = 1
times = times[times <= args.tmax]

# Sensor selection
region = []
if 'O' in args.sensors:
    region += ['Occipital']
if 'P' in args.sensors:
    region += ['Parietal']
if 'T' in args.sensors:
    region += ['Temporal']
if 'C' in args.sensors:
    region += ['Central']
if 'F' in args.sensors:
    region += ['Frontal']


# =============================================================================
# Load BERG's encoding model
# =============================================================================
# Load the encoding model
model = berg.get_encoding_model(
    args.encoding_model,
    subject=args.subject,
    selection={'region': region,
               'timepoints': timepoints}
    )


# =============================================================================
# Generate the in silico MEG responses
# =============================================================================
meg, metadata = berg.encode(model, images, return_metadata=True)


# =============================================================================
# Create the MEG RDM (pairwise decoding)
# =============================================================================
# The code assumes MEG responses in the format:
# (Image conditions × Repeats × Sensors × Time points)

# Results array of shape:
# (Image conditions × Image conditions × MEG time points)
meg_rdm = np.zeros((len(meg), len(meg), len(times)), dtype=np.float32)

# Loop over MEG time points and images
for t in tqdm(range(len(times))):
    for i1 in range(len(meg)):
        for i2 in range(i1):

            # Select the image condition data
            meg_cond_1 = meg[i1,:,:,t]
            meg_cond_2 = meg[i2,:,:,t]

            # SVM target vectors
            y_train = np.zeros(((len(meg_cond_1)-1)*2))
            y_train[int(len(y_train)/2):] = 1
            y_test = np.asarray((0, 1))
            scores = np.zeros(len(meg_cond_1))

            # Loop across repeats (leave-one-repeat-out cross-decoding)
            for r in range(len(meg_cond_1)):

                # Define the train/test partitions
                X_train = np.append(np.delete(meg_cond_1, r, 0),
                    np.delete(meg_cond_2, r, 0), 0)
                X_test = np.append(np.expand_dims(meg_cond_1[r], 0),
                    np.expand_dims(meg_cond_2[r], 0), 0)

                # Train the classifier
                dec_svm = SVC(kernel='linear')
                dec_svm.fit(X_train, y_train)

                # Test the classifier
                y_pred = dec_svm.predict(X_test)
                scores[r] = sum(y_pred == y_test) / len(y_test)

            # Store the accuracy
            meg_rdm[i1,i2,t] = np.mean(scores)
            meg_rdm[i2,i1,t] = meg_rdm[i1,i2,t]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'meg_rdm': meg_rdm,
    'metadata': metadata,
    'times': times
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'meg', 'dnn_layerwise_modeling', 'meg_rdms', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'meg_rdms_sub-' + format(args.subject, '02') + '_sensors-' + \
    '-'.join(args.sensors) + '.npy'

np.save(os.path.join(save_dir, file_name), results)