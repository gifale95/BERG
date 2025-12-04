def load_rsms(args, cv_subject=1, split='test'):
	"""Load the subject-average RSMs of the selected time points.

	Parameters
	----------
	args : Namespace
		Input arguments.
	cv_subject : int
		If args.cv==1, what is the cross-validation subject.
	split : str
		If args.cv==1, whether to use the 'train' (the RSMs averaged across N-1
		subjects), or 'test' (the RSMs of the remaining subject) RSMs.

	Returns
	-------
	rsm_1 : float
		First time point RSM.
	rsm_2 : float
		Second time point RSM.

	"""

	import os
	import numpy as np

	### ROI names ###
	idx = args.time_pair.find('-')
	time_1 = args.time_pair[:idx]
	time_2 = args.time_pair[idx+1:]

	### Load the RSMs ###
	dir_rsm = os.path.join(args.berg_dir, 'rnc_eeg', 'multivariate_rnc',
		'rsms')
	if args.cv == 0:
		file_name_time_1 = 'averaged_rsm_time-' + time_1 + \
			'_all_subjects.npy'
		file_name_time_2 = 'averaged_rsm_time-' + time_2 + \
			'_all_subjects.npy'
	elif args.cv == 1:
		file_name_time_1 = 'averaged_rsm_time-' + time_1 + '_cv_subject-' + \
			format(cv_subject, '02') + '_' + split + '.npy'
		file_name_time_2 = 'averaged_rsm_time-' + time_2 + '_cv_subject-' + \
			format(cv_subject, '02') + '_' + split + '.npy'

	rsm_1 = np.load(os.path.join(dir_rsm, file_name_time_1))
	rsm_2 = np.load(os.path.join(dir_rsm, file_name_time_2))

	### Output ###
	return rsm_1, rsm_2


def create_batches(n_batches, n_images_per_batch, n_used_images):
	"""Create random image batches.

	Parameters
	----------
	n_batches : int
		Initial number of image batches at each genetic optimization generation.
	n_images_per_batch : int
		Number of images per batch.
	n_used_images : int
		Total amount of used images.

	Returns
	-------
	image_batches : int
		Random image batches.

	"""

	import numpy as np
	from sklearn.utils import resample

	### Create image batches ###
	image_batches = np.zeros((n_batches,n_images_per_batch), dtype=int)
	for b in range(n_batches):
		image_batches[b,:] = resample(np.arange(n_used_images), replace=False,
			n_samples=n_images_per_batch)

	### Output ###
	return image_batches


def mutate(image_batches, n_used_images):
	"""Five mutated versions are created for each image batch, where for each
	version a different amount of images (1, 10%, 25%, 50%, and 75%) is
	randomly replaced with other images from the time point RSMs, while
	ensuring that no image is repeated within the same batch.

	Parameters
	----------
	image_batches : int
		Image batches that are mutated.
	n_used_images : int
		Total amount of used images.

	Returns
	-------
	mutated_image_batches : int
		Mutated image batches.

	"""

	import numpy as np
	from sklearn.utils import resample
	from copy import copy

	### Establish how many images will be mutated ###
	mutated_images = []
	mutated_images.append(1)
	mutated_images.append(round(image_batches.shape[1] / 100 * 10))
	mutated_images.append(round(image_batches.shape[1] / 100 * 25))
	mutated_images.append(round(image_batches.shape[1] / 100 * 50))
	mutated_images.append(round(image_batches.shape[1] / 100 * 75))
	mutated_images = np.asarray(mutated_images)

	### Mutate the image batches ###
	mutated_image_batches = []
	for b in range(image_batches.shape[0]):
		# Select new images not already present in the image batch
		new_imgs_pool = np.isin(np.arange(n_used_images), image_batches[b],
			assume_unique=True, invert=True)
		for m in range(len(mutated_images)):
			# Randomly select the images which will be mutated
			mutated_imgs_idx = resample(np.arange(len(image_batches[b])),
				replace=False, n_samples=mutated_images[m])
			# Randomly select the new images
			new_imgs = resample(np.where(new_imgs_pool)[0], replace=False,
				n_samples=mutated_images[m])
			# Replace the images
			mutated_chromosome = copy(image_batches[b])
			mutated_chromosome[mutated_imgs_idx] = new_imgs
			mutated_image_batches.append(mutated_chromosome)
	mutated_image_batches = np.asarray(mutated_image_batches)

	### Output ###
	return mutated_image_batches


def evaluate(image_batches, rsm_1, rsm_2):
	"""Use the image conditions from the image batches to compare, through RSA,
	the RSMs of the two time points. This will result in one correlation score
	for each image batch, indicating how well each batch aligns (or
	disentangles) the RSMs of the two time points.

	Parameters
	----------
	image_batches : int
		Matrix with image batches used for RSA, where the rows correspond to
		the different batches, and the columns to the different images per
		batch.
	rsm_1 : float
		First time point RSM.
	rsm_2 : float
		Second time point RSM.

	Returns
	-------
	scores : float
		RSA scores for each image batch.

	"""

	import numpy as np
	from scipy.stats import pearsonr

	### Empty results variable ###
	scores = np.zeros((len(image_batches)))
	idx_lower_tr = np.tril_indices(image_batches.shape[1], -1)

	### Score the image batches ###
	for c in range(len(image_batches)):
		vect_1 = rsm_1[image_batches[c]]
		vect_1 = vect_1[:,image_batches[c]]
		vect_2 = rsm_2[image_batches[c]]
		vect_2 = vect_2[:,image_batches[c]]
		vect_1 = vect_1[idx_lower_tr]
		vect_2 = vect_2[idx_lower_tr]
		scores[c] = pearsonr(vect_1, vect_2)[0]

	### Output ###
	return scores


def select(control_condition, image_batches, image_batches_scores, n_batches):
	"""Select and retain the best image batches, based on their RSA scores. To
	align the two time points, the image batches leading to highest correlation
	score (i.e., r=1) are kept. To disentangle the two time points, the image
	batches leading to lowest absolute correlation score (i.e, r=0) are kept.

	Parameters
	----------
	control_condition : str
		Whether to 'align' or 'disentangle' the multivariate fMRI responses for
		the two time points being compared.
	image_batches : int
		Image batches.
	image_batches_scores : float
		RSA score of each image batch.
	n_batches : int
		Initial number of image batches at each genetic optimization generation.

	Returns
	-------
	image_batches : int
		Selected image batches.
	image_batches_scores : float
		RSA scores of the selected image batches.

	"""

	import numpy as np

	### Select best image batches ###
	if control_condition == 'align':
		idx_best = np.argsort(image_batches_scores)[::-1]
	elif control_condition == 'disentangle':
		idx_best = np.argsort(abs(image_batches_scores))
	image_batches = image_batches[idx_best][:n_batches]
	image_batches_scores = image_batches_scores[idx_best][:n_batches]

	### Output ###
	return image_batches, image_batches_scores