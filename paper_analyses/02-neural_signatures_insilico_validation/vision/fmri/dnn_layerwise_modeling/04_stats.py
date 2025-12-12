"""Compute the significance of the RSA analysis between in silico fMRI
responses and DNN layerwise features.

Parameters
----------
subjects : list
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 8.
model : str
    Name of deep neural network model used to extract the image features.
    Available options are 'alexnet' and 'resnet50'.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=int)
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> RSA stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the RSA results
# =============================================================================
lh_rsa = {}
rh_rsa = {}

for s, sub in enumerate(args.subjects):
    for hemi in ['lh', 'rh']:

        results_dir = os.path.join(args.berg_dir,
            'neural_signatures_insilico_validation', 'vision', 'fmri',
            'dnn_layerwise_modeling', 'rsa', 'rsa_sub-'+format(sub, '02')+
            '_'+hemi+'_model-'+args.model+'.npy')
        results = np.load(results_dir, allow_pickle=True).item()

        for key, val in results['rsa'].items():
            if hemi == 'lh':
                if s == 0:
                    lh_rsa[key] = []
                lh_rsa[key].append(val)
            elif hemi == 'rh':
                if s == 0:
                    rh_rsa[key] = []
                rh_rsa[key].append(val)

for key in lh_rsa.keys():
    lh_rsa[key] = np.array(lh_rsa[key])
    rh_rsa[key] = np.array(rh_rsa[key])


# =============================================================================
# Compute the significance
# =============================================================================
# Significance threshold set by a two-tailed t-test across participants (N = 8)
# with Benjamini–Hochberg false discovery rate (FDR) correction; P = 0.05

# Empty result dictionaries
sig_lh_rsa = {}
sig_rh_rsa = {}
pval_corrected_lh_rsa = {}
pval_corrected_rh_rsa = {}

# Loop across model layers
for key in lh_rsa.keys():

    # Compute the p-values with t-test
    pval_lh_rsa = ttest_1samp(lh_rsa[key], 0, axis=0,
        alternative='two-sided')[1]
    pval_rh_rsa = ttest_1samp(rh_rsa[key], 0, axis=0,
        alternative='two-sided')[1]

    # Correct for multiple comparisons
    pval_all = np.append(pval_lh_rsa, pval_rh_rsa)
    sig, pval_corrected, _, _ = multipletests(pval_all, 0.05, 'fdr_bh')

    # Split the significance results into hemispheres
    sig_lh_rsa[key] = sig[:len(sig//2)]
    sig_rh_rsa[key] = sig[len(sig//2):]
    pval_corrected_lh_rsa[key] = pval_corrected[:len(sig//2)]
    pval_corrected_rh_rsa[key] = pval_corrected[len(sig//2):]

    # Delete unused variables
    del pval_lh_rsa, pval_rh_rsa, pval_all, sig, pval_corrected


# =============================================================================
# Color code the vertices based on the layer leading to highest RSA scores
# (Guclu & van Gerven, 2015, Fig. 4A; Lin et al., 2024, Fig. 6-7). # !!!
# =============================================================================



# =============================================================================
# Plot/report the layer assignment averaged across all vertices within all ROIs
# (V1, V2, V3, hV4, ventral) (Guclu & van Gerven, 2015, Fig. 4B).
# Also compute CIs. # !!!
# =============================================================================





# =============================================================================
# Save the results # !!!
# =============================================================================
results = {
    'sig_lh_rsa': sig_lh_rsa,
    'sig_rh_rsa': sig_rh_rsa,
    'pval_corrected_lh_rsa': pval_corrected_lh_rsa,
    'pval_corrected_rh_rsa': pval_corrected_rh_rsa,
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'dnn_layerwise_modeling', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats_model-' + args.model + '.npy'

np.save(os.path.join(save_dir, file_name), results)