"""Correlate the encoding accuracy of different encoding models with the
neural signature validation scores of their in silico EEG responses.

Parameters
----------
encoding_models : list
    List of BERG's encoding models used for generating the in silico EEG
    responses.
eeg_subjects : list
    List containing the subject identifiers for the EEG encoding models. Since
    the used encoding models are trained on THINGS EEG2, valid subject
    identifiers are integers from 1 to 10.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models', type=list, default=['eeg-things_eeg_2-alexnet_untrained', 'eeg-things_eeg_2-alexnet', 'eeg-things_eeg_2-vit_b_32'])
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Encoding model comparison <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the encoding accuracy scores
# =============================================================================
# Load the encoding accuracy scores
results_dir = os.path.join(args.berg_dir,
    'neural_signatures_insilico_validation', 'vision', 'eeg',
    'encoding_accuracy', 'stats', 'stats.npy')
results = np.load(results_dir, allow_pickle=True).item()
correlation = results['correlation']
metadata = results['metadata']

# Average the results across occipital and parietal channels, and across time
# points from 60ms after stimulus onset
encoding_accuracy = []
idx_time = np.where(metadata[0]['eeg']['times'] >= 0.06)[0]
for model in args.encoding_models:
    model = model[17:]
    encoding_accuracy.append(np.mean(
        correlation[model][:,:2,idx_time], (1, 2)))
encoding_accuracy = np.array(encoding_accuracy)


# =============================================================================
# Load the neural signature in silico validation scores
# =============================================================================
insilico_validation_scores = {}

for m, model in enumerate(args.encoding_models):

    # ERPs
    if m == 0:
        insilico_validation_scores['mse_erps'] = []
    results_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validatio   n', 'vision', 'eeg', 'erps',
        'erps', model, 'erps.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['mse_erps'].append(np.array(
        results['mse_erps']))

    # N170 faces
    if m == 0:
        insilico_validation_scores['erp_diff_avg'] = []
    results_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg', 'n170_faces',
        'stats', model, 'stats_channels-P7-P8-PO7-PO8-TP7-TP8.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['erp_diff_avg'].append(np.array(
        results['erp_diff_avg']))

   # Object categorization # !!!
    if m == 0:
        insilico_validation_scores['animate_selective_rois_animals'] = []
    results_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'object_categorization', 'stats', model, 'stats_channels-O-P.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['???'].append(np.array(results['???']))

   # DNN layerwise modeling
    if m == 0:
        insilico_validation_scores['rsa_peak_latency_dnn_layer_corr'] = []
    results_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'dnn_layerwise_modeling', 'stats', args.encoding_model,
        'stats_channels-O-P_dnn_model-alexnet.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['rsa_peak_latency_dnn_layer_corr'].append(
        np.array(results['rsa_peak_latency_dnn_layer_corr']))

   # LLM modeling
    if m == 0:
        insilico_validation_scores['diff_llm_rsa_late_early'] = []
    results_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'llm_modeling', 'stats', model, 'stats_channels-O-P.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['diff_llm_rsa_late_early'].append(
        np.array(results['diff_rsa_late_early']))

   # Behavioral modeling
    if m == 0:
        insilico_validation_scores['diff_beh_rsa_late_early'] = []
    results_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'eeg',
        'behavioral_modeling', 'stats', model, 'stats_channels-O-P.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['diff_beh_rsa_late_early'].append(
        np.array(results['diff_rsa_late_early']))


# =============================================================================
# Correlate the encoding accuracy and neural signature validation scores
# =============================================================================
corr = {}

for key, val in insilico_validation_scores.items():

    if key in ['mse_erps', 'erp_diff_avg']: # !!! Also include object categorization?
        corr[key] = pearsonr(np.array(encoding_accuracy).flatten(),
            np.array(val).flatten(), alternative='less')
    else:
        corr[key] = pearsonr(np.array(encoding_accuracy).flatten(),
            np.array(val).flatten(), alternative='greater')


# =============================================================================
# Save the results
# =============================================================================
results = {
    'metadata': metadata,
    'encoding_models': args.encoding_models,
    'eeg_subjects': args.eeg_subjects,
    'encoding_accuracy': encoding_accuracy,
    'insilico_validation_scores': insilico_validation_scores,
    'corr': corr
    }

# Create the saving directory
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'eeg', 'encoding_model_comparison', 'stats')
os.makedirs(save_dir, exist_ok=True)

# Save the results
np.save(os.path.join(save_dir, 'stats.npy'), results)