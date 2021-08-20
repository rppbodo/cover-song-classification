import numpy as np
import utils_classification as uc

def get_tensor(csv_filename, n_models, classes, tracks, dataset):
	filenames = uc.load_filenames(csv_filename, n_models)
	labels = uc.gather_labels(classes, tracks)
	tensor = uc.build_tensor(dataset, filenames)
	return labels, tensor

def get_filters(classes, tracks, train_test_split):
	versions_sets, count_tracks = uc.compute_versions_sets(classes, tracks)
	new_count_tracks = uc.sort_sets_by_length(versions_sets, count_tracks)
	train_tracks, test_tracks = uc.train_test_split_heuristic(tracks, train_test_split, versions_sets, new_count_tracks)
	train_filter, test_filter = uc.compute_filters(tracks, train_tracks, test_tracks)
	return train_filter, test_filter

def get_samples_and_labels(labels, tensor, train_filter, test_filter):
	train_labels = labels[train_filter]
	test_labels = labels[test_filter]
	
	train_tensor = tensor[train_filter]
	test_tensor = tensor[test_filter]
	
	train_samples_0s, train_samples_1s = uc.gather_samples(train_tensor, train_labels)
	test_samples_0s, test_samples_1s = uc.gather_samples(test_tensor, test_labels)
	
	train_samples = np.concatenate((train_samples_1s, train_samples_0s))
	train_labels = np.array(len(train_samples_1s) * [1] + len(train_samples_0s) * [0])
	
	test_samples = np.concatenate((test_samples_1s, test_samples_0s))
	test_labels = np.array(len(test_samples_1s) * [1] + len(test_samples_0s) * [0])
	
	return train_samples, train_labels, test_samples, test_labels, test_samples_0s, test_samples_1s

