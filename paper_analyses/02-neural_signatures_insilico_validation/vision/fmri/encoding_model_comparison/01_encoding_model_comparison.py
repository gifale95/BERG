"""Correlate the encoding accuracy of different encoding models with the
neural signature validation scores of their in silico fMRI responses.

Parameters
----------
encoding_models : list
    List of BERG's encoding models used for generating the in silico fMRI
    responses in fsavarage space.
fmri_subjects : list
    List containing the subject identifiers for the fMRI encoding models. Since
    the used encoding models are trained on NSD data, valid subject identifiers
    are integers from 1 to 8.
ncsnr_threshold : float
    The threshold on the noise ceiling signal-to-noise ratio (NCSNR) for
    vertex selection.
encoding_threshold : float
    The threshold on the encoding models explained variance for vertex
    selection (in % units).
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models', type=list, default=['fmri-nsd_fsaverage-alexnet_untrained', 'fmri-nsd_fsaverage-alexnet', 'fmri-nsd_fsaverage-huze'])
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float) # 0.2
parser.add_argument('--encoding_threshold', default=0, type=float) # 0
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
    'neural_signatures_insilico_validation', 'vision', 'fmri',
    'encoding_accuracy', 'stats', 'stats.npy')
results = np.load(results_dir, allow_pickle=True).item()
correlation_nsdcore = results['correlation_nsdcore']
correlation_nsdsynthetic = results['correlation_nsdsynthetic']
metadata = results['metadata']

# Average the results across vertex from both hemispheres, and aggregate them
# across subjects and encoding models
encoding_accuracy_nsdcore = []
encoding_accuracy_nsdsynthetic = []
for model in args.encoding_models:
    model = model[19:]
    acc_core = []
    acc_synthetic = []
    for s, sub in enumerate(args.fmri_subjects):
        # Vertex selection based on the NCSNR and encoding accuracy thresholds
        lh_idx_ncsnr = metadata[s]['fmri']['lh_ncsnr'] >= args.ncsnr_threshold
        rh_idx_ncsnr = metadata[s]['fmri']['rh_ncsnr'] >= args.ncsnr_threshold
        lh_idx_encoding = metadata[s]['encoding_models']\
            ['lh_explained_variance_nsdcore'] >= args.encoding_threshold
        rh_idx_encoding = metadata[s]['encoding_models']\
            ['rh_explained_variance_nsdcore'] >= args.encoding_threshold
        lh_idx = np.logical_and(lh_idx_ncsnr, lh_idx_encoding)
        rh_idx = np.logical_and(rh_idx_ncsnr, rh_idx_encoding)
        acc_core.append(np.mean(np.append(
            correlation_nsdcore[model]['lh'][0][lh_idx],
            correlation_nsdcore[model]['rh'][0][rh_idx])))
        acc_synthetic.append(np.mean(np.append(
            correlation_nsdsynthetic[model]['lh'][0][lh_idx],
            correlation_nsdsynthetic[model]['rh'][0][rh_idx])))
    encoding_accuracy_nsdcore.append(np.array(acc_core))
    encoding_accuracy_nsdsynthetic.append(np.array(acc_synthetic))


# =============================================================================
# Load the neural signature in silico validation scores
# =============================================================================
insilico_validation_scores = {}

for m, model in enumerate(args.encoding_models):

    # Retinotopy
    if m == 0:
        insilico_validation_scores['corr_polar_angle_silico_vivo'] = []
        insilico_validation_scores['corr_eccentricity_silico_vivo'] = []
    results_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'fmri',
        'retinotopy', 'GRID_RES-40_PROBE_SIGMA-0.25_BG_VALUE-0.5', 'stats',
        model, 'stats.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['corr_polar_angle_silico_vivo'].append(
        np.array(results['corr_polar_angle_silico_vivo']))
    insilico_validation_scores['corr_eccentricity_silico_vivo'].append(
        np.array(results['corr_eccentricity_silico_vivo']))

    # HVC selectivity
    if m == 0:
        insilico_validation_scores['corr_tval_silico_vivo_faces'] = []
        insilico_validation_scores['corr_tval_silico_vivo_bodies'] = []
        insilico_validation_scores['corr_tval_silico_vivo_places'] = []
    results_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'fmri',
        'hvc_selectivity', 'stats', model, 'stats.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['corr_tval_silico_vivo_faces'].append(
        np.array(results['corr_tval_silico_vivo']['faces']))
    insilico_validation_scores['corr_tval_silico_vivo_bodies'].append(
        np.array(results['corr_tval_silico_vivo']['bodies']))
    insilico_validation_scores['corr_tval_silico_vivo_places'].append(
        np.array(results['corr_tval_silico_vivo']['places']))

   # Tripartite organization
    if m == 0:
        insilico_validation_scores['animate_selective_rois_animals'] = []
        insilico_validation_scores['scene_selective_rois_big_objects'] = []
    results_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'fmri',
        'tripartite_organization', 'stats', model,
        'stats_images-naturalistic.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['animate_selective_rois_animals'].append(
        np.array(results['vertex_overlap']['animate_selective_rois_animals']))
    insilico_validation_scores['scene_selective_rois_big_objects'].append(
        np.array(results['vertex_overlap']['scene_selective_rois_big_objects']))

   # DNN layerwise modeling
    if m == 0:
        insilico_validation_scores['corr_best_layer_hierarchy_score'] = []
    results_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'fmri',
        'dnn_layerwise_modeling', 'stats', model, 'stats_model-alexnet.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['corr_best_layer_hierarchy_score'].append(
        np.array(results['corr_best_layer_hierarchy_score']))

   # LLM modeling
    if m == 0:
        insilico_validation_scores['diff_llm_rsa_high_early'] = []
    results_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'fmri',
        'llm_modeling', 'stats', model, 'stats.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['diff_llm_rsa_high_early'].append(
        np.array(results['diff_rsa_high_early']))

   # Behavioral modeling
    if m == 0:
        insilico_validation_scores['diff_behavioral_rsa_high_early'] = []
    results_dir = os.path.join(args.berg_dir,
        'neural_signatures_insilico_validation', 'vision', 'fmri',
        'behavioral_modeling', 'stats', model, 'stats.npy')
    results = np.load(results_dir, allow_pickle=True).item()
    insilico_validation_scores['diff_behavioral_rsa_high_early'].append(
        np.array(results['diff_rsa_high_early']))


# =============================================================================
# Correlate the encoding accuracy and neural signature validation scores
# =============================================================================
corr_nsdcore = {}
corr_nsdsynthetic = {}

for key, val in insilico_validation_scores.items():

    corr_nsdcore[key] = pearsonr(np.array(encoding_accuracy_nsdcore).flatten(),
        np.array(val).flatten(), alternative='greater')
    corr_nsdsynthetic[key] = pearsonr(
        np.array(encoding_accuracy_nsdsynthetic).flatten(),
        np.array(val).flatten(), alternative='greater')


# =============================================================================
# Save the results
# =============================================================================
results = {
    'metadata': metadata,
    'encoding_models': args.encoding_models,
    'fmri_subjects': args.fmri_subjects,
    'encoding_accuracy_nsdcore': encoding_accuracy_nsdcore,
    'encoding_accuracy_nsdsynthetic': encoding_accuracy_nsdsynthetic,
    'insilico_validation_scores': insilico_validation_scores,
    'corr_nsdcore': corr_nsdcore,
    'corr_nsdsynthetic': corr_nsdsynthetic
    }

# Create the saving directory
save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'encoding_model_comparison', 'stats')
os.makedirs(save_dir, exist_ok=True)

# Save the results
np.save(os.path.join(save_dir, 'stats.npy'), results)