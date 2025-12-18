"""Use BERG to generate the in silico fMRI responses to food images, and test
for food selective areas.

Parameters
----------
encoding_model : str
    The name of the fMRI encoding model in BERG to use for generating the
    in silico fMRI responses in surface space.
subject : int
    Subject identifier for the fMRI encoding model. Since the used encoding
    models are trained on NSD data, valid subject identifiers are integers from
    1 to 8.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import numpy as np
import os
from PIL import Image
import torch
from berg import BERG
from tqdm import tqdm
import gc
import torch
from sklearn.linear_model import LinearRegression
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Generate in silico fMRI <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the stimulus images
# =============================================================================
# Image directories
img_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'food_selectivity', 'stimuli')
categories = ['food', 'body', 'face', 'house', 'word']

# Load and format the images
images = {}
for cat in tqdm(categories):
    img_cat = []
    img_list = os.listdir(os.path.join(img_dir, cat))
    img_list.sort()
    for img_name in img_list:
        img_path = os.path.join(img_dir, cat, img_name)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        img_cat.append(img)
    img_cat = np.array(img_cat)
    img_cat = np.swapaxes(img_cat, 1, 3)  # BHWC to BCHW
    images[cat] = img_cat
    del img_cat


# =============================================================================
# Generate the in silico fMRI responses using BERG
# =============================================================================
# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Load the encoding model
model = berg.get_encoding_model(args.encoding_model, subject=args.subject)

# Loop across image categories
lh_insilico_fmri = {}
rh_insilico_fmri = {}
for c, cat in enumerate(categories):

    # Generate the in silico fMRI responses
    fmri, metadata = berg.encode(model, images[cat],
        return_metadata=True)

    # Store the in silico fMRI responses
    lh_insilico_fmri[cat] = fmri[0].astype(np.float32)
    rh_insilico_fmri[cat] = fmri[1].astype(np.float32)

    # Delete unused variables
    del fmri
    torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# Create the stimulus design matrix
# =============================================================================
n_cat = len(categories)
img_per_cat = 82
X = []

for c in range(n_cat):
    X_cat = np.zeros((1, n_cat), dtype=np.int8)
    X_cat[0,c] = 1
    X.append(np.repeat(X_cat, img_per_cat, 0))

X = np.concatenate(X)


# =============================================================================
# Format the in silico fMRI responses for modeling
# =============================================================================
lh_fmri = []
rh_fmri = []

for cat in categories:
    lh_fmri.append(lh_insilico_fmri[cat])
    rh_fmri.append(rh_insilico_fmri[cat])

lh_fmri = np.concatenate(lh_fmri)
rh_fmri = np.concatenate(rh_fmri)


# =============================================================================
# Model the in silico fMRI responses # !!! DELETE
# =============================================================================
# # Build the vertex-wise encoding models, and get their weights
# reg_lh = LinearRegression().fit(X, lh_fmri)
# reg_rh = LinearRegression().fit(X, rh_fmri)
# w_lh = reg_lh.coef_
# w_rh = reg_rh.coef_

# # For each vertex, compute the difference between its weight for the food
# # category, and the average weights for all other categories

# t_stat_num_lh = w_lh[:,0] - np.mean(w_lh[:,1:], 1)
# t_stat_num_rh = w_rh[:,0] - np.mean(w_rh[:,1:], 1)

# c = np.array([1, -0.25, -0.25, -0.25, -0.25])
# t_stat_num_lh_new = np.dot(w_lh, c)
# t_stat_num_rh_new = np.dot(w_rh, c)


# # Compute the standard error
# mse_lh = np.mean(np.square(reg_lh.predict(X) - lh_fmri), 0)
# mse_rh = np.mean(np.square(reg_rh.predict(X) - rh_fmri), 0)

# t_stat_denom = np.multiply(np.abs(mse),np.sqrt(np.linalg.multi_dot([c.T, xTx_pinv, c])))

# from numpy.linalg import pinv
# xTx_pinv = pinv(np.dot(X.T,X))
# w = np.dot(xTx_pinv, np.dot(X.T, lh_fmri))

# w_lh_new = w.T

# # Compute the t-statistics
# t_stat_lh = np.divide(t_stat_num_lh, t_stat_denom_lh)
# t_stat_rh = np.divide(t_stat_num_rh, t_stat_denom_rh)

# # Convert the t-statistics to p-values
# p_statistic_lh = 1 - tdistribution.cdf(t_stat_lh, 1000)
# p_statistic_rh = 1 - tdistribution.cdf(t_stat_rh, 1000)

# # Correct for multiple comparisons
# p_statistic_lh_sig, p_statistic_lh_corrected, _, _ = multipletests(
#     p_statistic_lh, 0.05, 'fdr_bh')
# p_statistic_rh_sig, p_statistic_rh_corrected, _, _ = multipletests(
#     p_statistic_rh, 0.05, 'fdr_bh')


# =============================================================================
# Model the in silico fMRI responses
# =============================================================================
# X: (N_images, K_labels)
# Y: (N_images, N_voxels)
# category_idx: index of category A

def fit_ols(X, Y):
    """
    Fits voxel-wise OLS models.

    Parameters
    ----------
    X : (N, K) design matrix
    Y : (N, V) voxel responses

    Returns
    -------
    beta_hat : (K, V) regression coefficients
    sigma2_hat : (V,) residual variance estimates
    XtX_inv : (K, K) inverse of X'X
    df : degrees of freedom
    """
    N, K = X.shape
    _, V = Y.shape

    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)

    beta_hat = XtX_inv @ X.T @ Y            # (K, V)
    residuals = Y - X @ beta_hat            # (N, V)

    df = N - K
    sigma2_hat = np.sum(residuals**2, axis=0) / df  # (V,)

    return beta_hat, sigma2_hat, XtX_inv, df

def category_vs_others_contrast(K, category_idx):
    """
    Builds contrast vector comparing category A vs. mean of others.
    """
    c = np.full(K, -1.0 / (K - 1))
    c[category_idx] = 1.0
    return c

def compute_t_statistics(beta_hat, sigma2_hat, XtX_inv, c):
    """
    Computes voxel-wise t-statistics for the contrast.
    """
    # Contrast estimate: c^T beta
    theta_hat = c @ beta_hat                # (V,)

    # Variance of contrast
    contrast_var = sigma2_hat * (c @ XtX_inv @ c)
    contrast_se = np.sqrt(contrast_var)

    t_values = theta_hat / contrast_se
    return t_values

def compute_p_values(t_values, df):
    """
    One-sided p-values for H1: contrast > 0
    """
    p_values = 1.0 - t.cdf(t_values, df)
    return p_values

# OLS estimation
beta_hat_lh, sigma2_hat_lh, XtX_inv_lh, df_lh = fit_ols(X, lh_fmri)
beta_hat_rh, sigma2_hat_rh, XtX_inv_rh, df_rh = fit_ols(X, rh_fmri)

# Construct contrast vector c
c = category_vs_others_contrast(K=X.shape[1], category_idx=0)

# Compute t-statistics
t_values_lh = compute_t_statistics(beta_hat_lh, sigma2_hat_lh, XtX_inv_lh, c)
t_values_rh = compute_t_statistics(beta_hat_rh, sigma2_hat_rh, XtX_inv_rh, c)

# Convert t-values to p-values
p_values_lh = compute_p_values(t_values_lh, df_lh)
p_values_rh = compute_p_values(t_values_rh, df_rh)

# Correct for multiple comparisons
p_values_lh_sig, p_values_lh_corrected, _, _ = multipletests(p_values_lh, 0.05,
    'fdr_bh')
p_values_rh_sig, p_values_rh_corrected, _, _ = multipletests(p_values_rh, 0.05,
    'fdr_bh')


# =============================================================================
# Save the results
# =============================================================================
results = {
    'lh_insilico_fmri': lh_insilico_fmri,
    'rh_insilico_fmri': rh_insilico_fmri,
    'metadata': metadata,
    'lh_tval': t_values_lh,
    'rh_tval': t_values_rh,
    'lh_pval': p_values_lh_corrected,
    'rh_pval': p_values_rh_corrected,
    'lh_sig': p_values_lh_sig,
    'rh_sig': p_values_rh_sig
    }

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'food_selectivity', 'single_subject_results')
os.makedirs(save_dir, exist_ok=True)

file_name = 'results_sub-' + format(args.subject, '02') + '.npy'

np.save(os.path.join(save_dir, file_name), results)