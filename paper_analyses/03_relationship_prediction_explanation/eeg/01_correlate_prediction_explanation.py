"""Correlate the prediction accuracy of different encoding models with the
explanation accuracy scores of their in silico EEG responses.

Parameters
----------
encoding_models : list
    List of BERG's encoding models used for generating the in silico EEG
    responses.
eeg_subjects : list
    List containing the subject identifiers for the EEG encoding models. Since
    the used encoding models are trained on THINGS EEG2, valid subject
    identifiers are integers from 1 to 10.
n_iter : int
    Amount of iterations for the statistical significance permutation test.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.utils import resample
from tqdm import tqdm
import random

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models', type=list, default=['eeg-things_eeg_2-alexnet_untrained', 'eeg-things_eeg_2-alexnet', 'eeg-things_eeg_2-vit_b_32'])
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Correlation between prediction and explanation accuracy <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the prediction accuracy scores
# =============================================================================
# Load the prediction accuracy scores
results_dir = os.path.join(args.berg_dir, 'prediction_accuracy_noise_analysis',
    'eeg', 'stats', 'stats.npy')
results = np.load(results_dir, allow_pickle=True).item()
correlation = results['correlation']
metadata = results['metadata']

# Average the results across occipital and parietal channels, and across time
# points from 60ms after stimulus onset
prediction_accuracy = []
idx_time = np.where(metadata[0]['eeg']['times'] >= 0.06)[0]
for model in args.encoding_models:
    model = model[17:]
    prediction_accuracy.append(np.mean(correlation[model][:,idx_time], 1))
prediction_accuracy = np.array(prediction_accuracy)


# =============================================================================
# Load the explanation accuracy scores
# =============================================================================
insilico_validation_scores = {}

for m, model in enumerate(args.encoding_models):

    # N170 faces
    if m == 0:
        insilico_validation_scores['erp_diff_avg'] = []
    results_dir = os.path.join(args.berg_dir,
        'insilico_capture_of_neural_signatures', 'eeg', 'n170_faces',
        'stats', model, 'stats_channels-P7-P8-PO7-PO8-TP7-TP8.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['erp_diff_avg'].append(np.array(
        results['erp_diff_avg']))

   # Object decoding dynamics
    if m == 0:
        insilico_validation_scores['decoding_peaks_diff'] = []
    results_dir = os.path.join(args.berg_dir,
        'insilico_capture_of_neural_signatures', 'eeg',
        'object_decoding_dynamics', 'stats', model, 'stats_channels-O-P.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['decoding_peaks_diff'].append(np.array(
        results['decoding_peaks_diff']))

   # DNN layerwise modeling
    if m == 0:
        insilico_validation_scores['corr_dnn_layer_eeg_times'] = []
    results_dir = os.path.join(args.berg_dir,
        'insilico_capture_of_neural_signatures', 'eeg',
        'dnn_layerwise_modeling', 'stats', model,
        'stats_channels-O-P_dnn_model-alexnet.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['corr_dnn_layer_eeg_times'].append(
        np.array(results['corr_dnn_layer_eeg_times']))

   # LLM modeling
    if m == 0:
        insilico_validation_scores['diff_llm_rsa_late_early'] = []
    results_dir = os.path.join(args.berg_dir,
        'insilico_capture_of_neural_signatures', 'eeg',
        'llm_modeling', 'stats', model, 'stats_channels-O-P.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['diff_llm_rsa_late_early'].append(
        np.array(results['diff_rsa_late_early']))

   # Behavioral modeling
    if m == 0:
        insilico_validation_scores['diff_beh_rsa_late_early'] = []
    results_dir = os.path.join(args.berg_dir,
        'insilico_capture_of_neural_signatures', 'eeg',
        'behavioral_modeling', 'stats', model, 'stats_channels-O-P.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['diff_beh_rsa_late_early'].append(
        np.array(results['diff_rsa_late_early']))


# =============================================================================
# Correlate the prediction and explanation accuracy scores
# =============================================================================
corr = {}

for key, val in insilico_validation_scores.items():

    corr[key] = pearsonr(prediction_accuracy.flatten(),
        np.array(val).flatten(), alternative='greater')


# =============================================================================
# Compute the significance through permutation tests
# =============================================================================
# Permutation null distribution empty arrays
null_dist = {}
for key in insilico_validation_scores.keys():
    null_dist[key] = np.zeros((args.n_iter))

# Loop across permutation iterations
for i in tqdm(range(args.n_iter)):

    # Permutation indices
    idx_pred = resample(np.arange(len(prediction_accuracy.flatten())),
        replace=False)
    idx_expl = resample(np.arange(len(prediction_accuracy.flatten())),
        replace=False)

    # Compute the correlation scores for the permuted data
    for key, val in insilico_validation_scores.items():
        null_dist[key][i] = pearsonr(
            prediction_accuracy.flatten()[idx_pred],
            np.array(val).flatten()[idx_expl],
            alternative='greater')[0]

# Compute the p-values based on the null distribution
p_values = {}
for key, val in insilico_validation_scores.items():
    p_values[key] = (np.sum(null_dist[key] >= corr[key][0]) + 1) / \
        (args.n_iter + 1)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'metadata': metadata,
    'encoding_models': args.encoding_models,
    'eeg_subjects': args.eeg_subjects,
    'prediction_accuracy': prediction_accuracy,
    'insilico_validation_scores': insilico_validation_scores,
    'corr': corr,
    'p_values': p_values
    }

# Create the saving directory
save_dir = os.path.join(args.berg_dir, 'relationship_prediction_explanation',
    'eeg', 'stats')
os.makedirs(save_dir, exist_ok=True)

# Save the results
np.save(os.path.join(save_dir, 'stats.npy'), results)