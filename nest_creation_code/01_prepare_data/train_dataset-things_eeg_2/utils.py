def epoching(args, data_part):
	"""This function first converts the EEG data to MNE raw format and
	performs channel selection, filtering, epoching, current source density
	transform, frequency downsampling and baseline correction. Then, it sorts
	the EEG data of each session based on the image conditions.

	Parameters
	----------
	args : Namespace
		Input arguments.
	data_part : str
		'test' or 'training' data partitions.

	Returns
	-------
	epoched_data : list of float
		Epoched EEG data.
	img_conditions : list of int
		Unique image conditions of the epoched and sorted EEG data.
	ch_names : list of str
		EEG channel names.
	times : float
		EEG time points.

	"""

	import os
	import mne
	import numpy as np

	### Loop across data collection sessions ###
	epoched_data = []
	img_conditions = []
	for s in range(args.n_ses):

		### Load the EEG data and convert it to MNE raw format ###
		eeg_dir = os.path.join('raw_data', 'sub-'+format(args.subject,'02'),
			'ses-'+format(s+1,'02'), 'raw_eeg_'+data_part+'.npy')
		eeg_data = np.load(os.path.join(args.things_eeg_2_dir, eeg_dir),
			allow_pickle=True).item()
		ch_names = eeg_data['ch_names']
		sfreq = eeg_data['sfreq']
		ch_types = eeg_data['ch_types']
		eeg_data = eeg_data['raw_eeg_data']
		# Convert to MNE raw format
		info = mne.create_info(ch_names, sfreq, ch_types)
		raw = mne.io.RawArray(eeg_data, info)
		del eeg_data

		### Filter the data ###
		if args.highpass != None or args.lowpass != None:
			raw = raw.copy().filter(l_freq=args.highpass, h_freq=args.lowpass)

		### Get events, drop unused channels and reject target trials ###
		events = mne.find_events(raw, stim_channel='stim')
		# Drop the 'stim' channel (keep only the 'eeg' channels)
		raw.pick_types(eeg=True, stim=False)
		# Select only occipital (O) channels
#		chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'],
#			'^O *|^P *'))
#		new_chans = [raw.info['ch_names'][c] for c in chan_idx]
#		raw.pick_channels(new_chans)
		# Reject the target trials (event 99999)
		idx_target = np.where(events[:,2] == 99999)[0]
		events = np.delete(events, idx_target, 0)
		img_cond = np.unique(events[:,2])

		### Epoching ###
		epochs = mne.Epochs(raw, events, tmin=args.tmin, tmax=args.tmax,
			baseline=None, preload=True)
		del raw

		### Compute current source density ###
		if args.csd == 1:
			# Load the EEG info dictionary from the source data, and extract
			# the channels positions
			eeg_dir = os.path.join(args.things_eeg_2_dir, 'source_data', 'sub-'+
				format(args.subject,'02'), 'ses-'+format(s+1,'02'), 'eeg',
				'sub-'+format(args.subject,'02')+'_ses-'+format(s+1,'02')+
				'_task-test_eeg.vhdr')
			source_eeg = mne.io.read_raw_brainvision(eeg_dir, preload=True)
			source_info = source_eeg.info
			del source_eeg
			# Create the channels montage file
			ch_pos = {}
			for c, dig in enumerate(source_info['dig']):
				if c > 2:
					if source_info['ch_names'][c-3] in epochs.info['ch_names']:
						ch_pos[source_info['ch_names'][c-3]] = dig['r']
			montage = mne.channels.make_dig_montage(ch_pos)
			# Apply the montage to the epoched data
			epochs.set_montage(montage)
			# Compute current source density
			epochs = mne.preprocessing.compute_current_source_density(epochs,
				lambda2=1e-05, stiffness=4)

		### Resample the epoched data ###
		if args.sfreq < 1000:
			epochs.resample(args.sfreq)
		ch_names = epochs.info['ch_names']
		times = epochs.times
		info = epochs.info
		epochs = epochs.get_data()

		### Baseline correction ###
		if args.baseline_correction == 1:
			epochs = mne.baseline.rescale(epochs, times, baseline=(None, 0),
				mode=args.baseline_mode)

		### Sort the data ###
		# Select only a maximum number of EEG repetitions
		if data_part == 'test':
			max_rep = 20
		else:
			max_rep = 2
		# Sorted data matrix of shape:
		# (Image conditions x EEG repetitions x EEG channels x EEG time points)
		sorted_data = np.zeros((len(img_cond),max_rep,epochs.shape[1],
			epochs.shape[2]))
		for i in range(len(img_cond)):
			# Find the indices of the selected image condition
			idx = np.where(events[:,2] == img_cond[i])[0]
			# Select only the max number of EEG repetitions
			idx = idx[:max_rep]
			sorted_data[i] = epochs[idx]
		del epochs
		epoched_data.append(sorted_data)
		img_conditions.append(img_cond)
		del sorted_data

	### Output ###
	return epoched_data, img_conditions, ch_names, times


def save_prepr(args, epoched_test, epoched_train, img_conditions_train,
	ch_names, times):
	"""Merge the EEG data of all sessions, and reshape it to the format:
	(Image conditions x EEG repetitions x EEG channels x EEG time points).
	Then, the data of both training and testing EEG partitions is saved.

	Parameters
	----------
	args : Namespace
		Input arguments.
	epoched_test : list of float
		Epoched test EEG data.
	epoched_train : list of float
		Epoched training EEG data.
	img_conditions_train : list of int
		Unique image conditions of the epoched and sorted train EEG data.
	ch_names : list of str
		EEG channel names.
	times : float
		EEG time points.
	ncsnr : float
		Noise ceiling SNR.

	"""

	import numpy as np
	import os
	import h5py

	### Load the image metadata ###
	metadata_dir = os.path.join(args.things_eeg_2_dir, 'image_set',
		'image_metadata.npy')
	img_metadata = np.load(metadata_dir, allow_pickle=True).item()
	test_img_info = {
		'test_img_concepts': img_metadata['test_img_concepts'],
		'test_img_concepts_THINGS': img_metadata['test_img_concepts_THINGS'],
		'test_img_files': img_metadata['test_img_files']
		}
	train_img_info = {
		'train_img_concepts': img_metadata['train_img_concepts'],
		'train_img_concepts_THINGS': img_metadata['train_img_concepts_THINGS'],
		'train_img_files': img_metadata['train_img_files']
		}

	### Save the metadata ###
	metadata = {
		'test_img_info': test_img_info,
		'train_img_info': train_img_info,
		'ch_names': ch_names,
		'times': times
		}
	save_dir = os.path.join(args.nest_dir, 'model_training_datasets',
		'train_dataset-things_eeg_2')
	file_name = 'metadata_subject-'+str(args.subject)+'.npy'
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	np.save(os.path.join(save_dir, file_name), metadata)

	### Merge and save the test data ###
	for s in range(args.n_ses):
		if s == 0:
			merged_test = epoched_test[s]
		else:
			merged_test = np.append(merged_test, epoched_test[s], 1)
	del epoched_test
	# Convert to float32
	merged_test = merged_test.astype(np.float32)
	# Save the data
	file_name_test = 'eeg_sub-'+format(args.subject,'02')+'_split-test.h5'
	with h5py.File(os.path.join(save_dir, file_name_test), 'w') as f:
		f.create_dataset('eeg', data=merged_test, dtype=np.float32)
	del merged_test

	### Merge and save the training data ###
	for s in range(args.n_ses):
		if s == 0:
			white_data = epoched_train[s]
			img_cond = img_conditions_train[s]
		else:
			white_data = np.append(white_data, epoched_train[s], 0)
			img_cond = np.append(img_cond, img_conditions_train[s], 0)
	del epoched_train, img_conditions_train
	# Data matrix of shape:
	# (Image conditions x EEG repetitions x EEG channels x EEG time points)
	merged_train = np.zeros((len(np.unique(img_cond)), white_data.shape[1]*2,
		white_data.shape[2],white_data.shape[3]))
	for i in range(len(np.unique(img_cond))):
		# Find the indices of the selected category
		idx = np.where(img_cond == i+1)[0]
		for r in range(len(idx)):
			if r == 0:
				ordered_data = white_data[idx[r]]
			else:
				ordered_data = np.append(ordered_data, white_data[idx[r]], 0)
		merged_train[i] = ordered_data
	# Convert to float32
	merged_train = merged_train.astype(np.float32)
	# Save the data
	file_name_train = 'eeg_sub-'+format(args.subject,'02')+'_split-train.h5'
	with h5py.File(os.path.join(save_dir, file_name_train), 'w') as f:
		f.create_dataset('eeg', data=merged_train, dtype=np.float32)
