"""Compute the significance of the RSA analysis between in silico fMRI
responses and behavioral embeddings.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses in fsavarage space.
subjects : list
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 8.
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
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> RSA stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the RSA results
# =============================================================================
lh_rsa = []
rh_rsa = []

for sub in args.subjects:
    for hemi in ['lh', 'rh']:

        results_dir = os.path.join(args.berg_dir,
            'neural_signatures_insilico_validation', 'vision', 'fmri',
            'behavioral_modeling', 'rsa', args.encoding_model, 'rsa_sub-'+
            format(sub, '02')+'_'+hemi+'.npy')
        results = np.load(results_dir, allow_pickle=True).item()

        if hemi == 'lh':
            lh_rsa.append(results['rsa'])
        elif hemi == 'rh':
            rh_rsa.append(results['rsa'])
        del results

lh_rsa = np.array(lh_rsa)
rh_rsa = np.array(rh_rsa)


# =============================================================================
# Compute the significance
# =============================================================================
# Significance threshold set by a two-tailed t-test across participants (N = 8)
# with Benjamini–Hochberg false discovery rate (FDR) correction; P = 0.05

# Compute the p-values with t-tests
pval_lh_rsa = ttest_1samp(lh_rsa, 0, axis=0, alternative='two-sided')[1]
pval_rh_rsa = ttest_1samp(rh_rsa, 0, axis=0, alternative='two-sided')[1]

# Correct for multiple comparisons
pval_all = np.append(pval_lh_rsa, pval_rh_rsa)
sig, pval_corrected, _, _ = multipletests(pval_all, 0.05, 'fdr_bh')

# Split the significance results into hemispheres
sig_lh_rsa = sig[:len(sig)//2]
sig_rh_rsa = sig[len(sig)//2:]
pval_corrected_lh_rsa = pval_corrected[:len(sig)//2]
pval_corrected_rh_rsa = pval_corrected[len(sig)//2:]


# =============================================================================
# Save the results
# =============================================================================
results = {
    'sig_lh_rsa': sig_lh_rsa,
    'sig_rh_rsa': sig_rh_rsa,
    'pval_corrected_lh_rsa': pval_corrected_lh_rsa,
    'pval_corrected_rh_rsa': pval_corrected_rh_rsa
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'behavioral_modeling', 'stats', args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)