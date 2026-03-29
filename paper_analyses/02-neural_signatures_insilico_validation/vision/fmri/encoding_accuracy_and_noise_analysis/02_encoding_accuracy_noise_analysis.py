"""Get the in-distribution (ID) and out-of-distribution (OOD) encoding accuracy
of BERG's fMRI encoding models trained on NSD.

This code additionally compares the noise of the in silico fMRI responses
(i.e., the fMRI responses generated from encoding models) with the noise of the
in vivo (i.e., target) responses from the NSD experiment, by comparing how
much variance can these two data types explain for a third, independent split
of NSD responses.

Because the in silico neural responses did not capture all signal variance for
the in vivo NSD responses, the in silico neural responses explaining more
variance than NSD's in-vivo responses would be indicative of the former being
less affected by noise.

The comparison is carried out through three predictions, using the in silico
and in vivo fMRI responses for the 515 test images. Each prediction involves
explaining single NSD in vivo response trials with a different predictor.
The first predictor consists of the two remaining NSD in vivo response trials,
each used independently. The evaluation is repeated until each of the three
trials is used as the target to be explained and the remaining two trials as
separate predictors, and the explained variance scores from the different
evaluations (N = 6 evaluations) are then averaged.
The second predictor consists of the average of the two remaining NSD in vivo
response trials. The evaluation is repeated until each of the three trials is
used as the target to be explained and the average of the remaining two trials
as predictor, and the explained variance scores from the different evaluations
(N = 3 evaluations) are then averaged.
The third predictor consists of the in silico responses from the trained
encoding models. The evaluation is repeated until each of the three trials is
used as the target to be explained by the same in silico responses, and the
explained variance scores from the different evaluations (N = 3 evaluations) is
then averaged.
These comparisons are carried out independently for each vertex and subject.

Parameters
----------
encoding_model : str
    The name of BERG's encoding model used for generating the in silico fMRI
    responses in fsavarage space.
subjects : list of int
    The subject identifiers for the fMRI encoding models. Since the used
    encoding models are trained on NSD data, valid subject identifiers are
    integers from 1 to 8.
hemispheres : list of str
    List of strings containing the hemispheres used for the analyses.
    Possible values  are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
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
from tqdm import tqdm
from berg import BERG
import h5py
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_model', type=str, default='fmri-nsd_fsaverage-huze')
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6, 7, 8], type=list)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--ncsnr_threshold', default=0.2, type=float) # 0.2
parser.add_argument('--encoding_threshold', default=0, type=float) # 0
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Encoding accuracy and noise analysis <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Loop across subjects and hemispheres
# =============================================================================
# Empty result variables
correlation_nsdcore = {}
correlation_nsdsynthetic = {}
metadata_berg = []
corr_iv1tr_is = {}
corr_iv1tr_iv1tr = {}
corr_iv1tr_iv2tr = {}

# Initialize BERG
berg = BERG(berg_dir=args.berg_dir)

# Loop across subjects and hemispheres
for s, sub in enumerate(tqdm(args.subjects)):
    for h, hemi in enumerate(args.hemispheres):

        if s == 0:
            correlation_nsdcore[hemi] = []
            correlation_nsdsynthetic[hemi] = []
            corr_iv1tr_is[hemi] = []
            corr_iv1tr_iv1tr[hemi] = []
            corr_iv1tr_iv2tr[hemi] = []


# =============================================================================
# Get the ID and OOD encoding accuracy of the in silico fMRI responses
# =============================================================================
        # Get the metadata
        metadata = berg.get_model_metadata(
            args.encoding_model,
            subject=sub
        )

        # Store the metadata
        if h == 0:
            metadata_berg.append(metadata)

        # Extract the encoding accuracy
        correlation_nsdcore[hemi].append(metadata['encoding_models']\
            [f'{hemi}_correlation_nsdcore'])
        correlation_nsdsynthetic[hemi].append(metadata['encoding_models']\
            [f'{hemi}_correlation_nsdsynthetic'])


# =============================================================================
# Load the in silico fMRI responses
# =============================================================================
        data_dir = os.path.join(args.berg_dir,
            'neural_signatures_insilico_validation',
            'vision', 'fmri', 'encoding_accuracy', 'insilico_fmri_responses',
            args.encoding_model, 'insilico_fmri_responses_sub-'+
            format(sub, '02')+'_'+hemi+'.npy')

        data = np.load(data_dir, allow_pickle=True).item()
        fmri_insilico = data['fmri'].astype(np.float32)
        fmri_insilico = np.nan_to_num(fmri_insilico)


# =============================================================================
# Load the in vivo fMRI responses
# =============================================================================
        # The in vivo fMRI responses were prepared using this code:
        # https://github.com/gifale95/BERG/blob/main/berg_creation_code/01_prepare_data/train_dataset-nsd_fsaverage/prepare_nsd_fsaverage.py

        # Data directories
        data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
            'train_dataset-nsd_fsaverage')
        fmri_dir = os.path.join(data_dir, f'{hemi}_betas_subject-{sub}.h5')
        metadata_dir = os.path.join(data_dir, f'metadata_subject-{sub}.npy')

        # Access the data
        fmri_invivo_all = h5py.File(fmri_dir, 'r')['betas']
        metadata_invivo = np.load(metadata_dir, allow_pickle=True).item()

        # Only select the fMRI responses for the 515 test images
        test_img_num = metadata_invivo['test_img_num']
        unique_test_img = np.unique(test_img_num)
        img_presentation_order = metadata_invivo['img_presentation_order']
        n_trials = 3
        fmri_invivo = np.zeros((len(unique_test_img), n_trials,
            fmri_invivo_all.shape[1]), dtype=np.float32)
        for i, img_num in enumerate(unique_test_img):
            img_presentation_idx = np.where(img_presentation_order == img_num)[0]
            fmri_invivo[i] = fmri_invivo_all[img_presentation_idx]
        fmri_invivo = np.nan_to_num(fmri_invivo)
        del metadata_invivo


# =============================================================================
# Correlate in silico and in vivo target fMRI for the noise analysis
# =============================================================================
        # Correlate in vivo target fMRI single trials with other in vivo single
        # trials, and with in silico fMRI responses
        # iv1tr => in vivo single trial
        # is => in silico
        comparisons = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
        corr_iv1tr_is_sub = np.zeros((len(comparisons), fmri_insilico.shape[1]))
        corr_iv1tr_iv1tr_sub = np.zeros((len(comparisons), 2, fmri_insilico.shape[1]))
        for c, comp in enumerate(comparisons):
            for v in range(fmri_insilico.shape[1]):
                corr_iv1tr_is_sub[c,v] = pearsonr(fmri_invivo[:,comp[0],v],
                    fmri_insilico[:,v])[0]
                corr_iv1tr_iv1tr_sub[c,0,v] = pearsonr(fmri_invivo[:,comp[0],v],
                    fmri_invivo[:,comp[1],v])[0]
                corr_iv1tr_iv1tr_sub[c,1,v] = pearsonr(fmri_invivo[:,comp[0],v],
                    fmri_invivo[:,comp[2],v])[0]

        # Correlate target fMRI single trials with the average of the two other
        # trials
        # iv1tr => in vivo single trial
        # iv2tr => in vivo averaged across two trials
        corr_iv1tr_iv2tr_sub = np.zeros((len(comparisons), fmri_insilico.shape[1]))
        for c, comp in enumerate(comparisons):
            for v in range(fmri_insilico.shape[1]):
                corr_iv1tr_iv2tr_sub[c,v] = pearsonr(fmri_invivo[:,comp[0],v],
                    np.mean(fmri_invivo[:,comp[1:],v], 1))[0]

        # Average the correlations across comparisons
        corr_iv1tr_is_sub = np.mean(corr_iv1tr_is_sub, 0)
        corr_iv1tr_iv1tr_sub = np.mean(corr_iv1tr_iv1tr_sub, (0, 1))
        corr_iv1tr_iv2tr_sub = np.mean(corr_iv1tr_iv2tr_sub, 0)

        # NCSNR and encoding accuracy vertex selection
        idx_ncsnr = metadata['fmri'][f'{hemi}_ncsnr'] >= \
            args.ncsnr_threshold
        encoding = metadata['encoding_models']\
            [hemi+'_explained_variance_nsdcore']
        idx_encoding = encoding >= args.encoding_threshold
        idx_nan = ~np.logical_and(idx_ncsnr, idx_encoding)
        corr_iv1tr_is_sub[idx_nan] = np.nan
        corr_iv1tr_iv1tr_sub[idx_nan] = np.nan
        corr_iv1tr_iv2tr_sub[idx_nan] = np.nan

        # Store the correlations
        corr_iv1tr_is[hemi].append(corr_iv1tr_is_sub)
        corr_iv1tr_iv1tr[hemi].append(corr_iv1tr_iv1tr_sub)
        corr_iv1tr_iv2tr[hemi].append(corr_iv1tr_iv2tr_sub)
        del fmri_insilico, fmri_invivo, corr_iv1tr_is_sub, \
            corr_iv1tr_iv1tr_sub, corr_iv1tr_iv2tr_sub


# =============================================================================
# Average the noise analysis results across vertices from both hemispheres
# =============================================================================
corr_iv1tr_is_avg = []
corr_iv1tr_iv1tr_avg = []
corr_iv1tr_iv2tr_avg = []
for s in range(len(args.subjects)):
    corr_iv1tr_is_sub = np.concatenate(
        [corr_iv1tr_is[hemi][s] for hemi in args.hemispheres])
    corr_iv1tr_iv1tr_sub = np.concatenate(
        [corr_iv1tr_iv1tr[hemi][s] for hemi in args.hemispheres])
    corr_iv1tr_iv2tr_sub = np.concatenate(
        [corr_iv1tr_iv2tr[hemi][s] for hemi in args.hemispheres])
    corr_iv1tr_is_avg.append(np.nanmean(corr_iv1tr_is_sub))
    corr_iv1tr_iv1tr_avg.append(np.nanmean(corr_iv1tr_iv1tr_sub))
    corr_iv1tr_iv2tr_avg.append(np.nanmean(corr_iv1tr_iv2tr_sub))
    del corr_iv1tr_is_sub, corr_iv1tr_iv1tr_sub, corr_iv1tr_iv2tr_sub
corr_iv1tr_is_avg = np.array(corr_iv1tr_is_avg)
corr_iv1tr_iv1tr_avg = np.array(corr_iv1tr_iv1tr_avg)
corr_iv1tr_iv2tr_avg = np.array(corr_iv1tr_iv2tr_avg)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'correlation_nsdcore': correlation_nsdcore,
    'correlation_nsdsynthetic': correlation_nsdsynthetic,
    'metadata': metadata_berg,
    'corr_iv1tr_is': corr_iv1tr_is,
    'corr_iv1tr_iv2tr': corr_iv1tr_iv2tr,
    'corr_iv1tr_iv1tr': corr_iv1tr_iv1tr,
    'corr_iv1tr_is_avg': corr_iv1tr_is_avg,
    'corr_iv1tr_iv2tr_avg': corr_iv1tr_iv2tr_avg,
    'corr_iv1tr_iv1tr_avg': corr_iv1tr_iv1tr_avg
}

save_dir = os.path.join(args.berg_dir, 'neural_signatures_insilico_validation',
    'vision', 'fmri', 'encoding_accuracy', 'encoding_accuracy',
    args.encoding_model)
os.makedirs(save_dir, exist_ok=True)

file_name = 'encoding_accuracy.npy'

np.save(os.path.join(save_dir, file_name), results)