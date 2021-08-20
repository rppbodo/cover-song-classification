import os
import csv
import numpy as np
from itertools import product

def load_filenames(csv_filename, n_models):
	csv_file = open(csv_filename)
	csv_reader = csv.reader(csv_file, delimiter=",")
	rows = []
	for row in csv_reader:
		rows.append("-".join(row) + ".npz")
		if len(rows) == n_models:
			break
	csv_file.close()
	return np.array(rows)

def gather_labels(classes, tracks):
	n = len(tracks)
	labels = np.random.random((n, n, 1))
	for i in range(n):
		for j in range(n):
			if tracks[i].class_ == tracks[j].class_:
				labels[i,j] = 1
			else:
				labels[i,j] = 0
	return labels

def build_tensor(dataset, filenames):
	tensor = None
	for filename in filenames:
		similarity_matrix_fp = np.load(os.path.join(dataset, "results", filename))
		similarity_matrix = similarity_matrix_fp["arr_0"]
		
		if tensor is None:
			tensor = similarity_matrix.reshape((similarity_matrix.shape[0], similarity_matrix.shape[1], 1))
		else:
			tensor = np.append(tensor, similarity_matrix.reshape((similarity_matrix.shape[0], similarity_matrix.shape[1], 1)), axis=2)
		
		similarity_matrix_fp.close()
	return tensor

def compute_versions_sets(classes, tracks):
	# building individual sets with all versions of a track
	versions_sets = {}
	
	for class_ in classes:
		versions_sets[class_] = []
	
	for track in tracks:
		versions_sets[track.class_].append(track)
	
	# computing histogram of versions per original track
	histogram = []
	count_tracks = []
	for key in versions_sets.keys():
		histogram.append(len(versions_sets[key]))
		count_tracks.append([key, len(versions_sets[key])])
	
	return versions_sets, np.array(count_tracks)

def sort_sets_by_length(versions_sets, count_tracks):
	# defining split points to separate versions sets with same number of tracks
	split_points = []
	indexes = np.argsort(count_tracks[:,1])[::-1] # sort versions sets by number of tracks
	count_tracks = count_tracks[indexes]
	last = None
	for i in range(len(count_tracks)):
		if count_tracks[i][1] != last:
			split_points.append(i)
			last = count_tracks[i][1]
	split_points.append(len(count_tracks))
	
	# shuffle versions sets with same number of tracks
	new_count_tracks = None
	for i in range(len(split_points)-1):
		start = split_points[i]
		end = split_points[i+1]
		
		assert len(set([cs[1] for cs in count_tracks[start:end]])) == 1
		
		items = count_tracks[start:end]
		np.random.shuffle(items)
		
		if new_count_tracks is None:
			new_count_tracks = items
		else:
			new_count_tracks = np.concatenate((new_count_tracks, items)) # TODO: test this with covers80
	
	return new_count_tracks

def train_test_split_heuristic(tracks, train_test_split, versions_sets, count_tracks):
	# split train test heuristic
	train_tracks = []
	test_tracks = []
	while len(count_tracks) > 0:
		current_class = count_tracks[0]
		
		n_samples = len(train_tracks) ** 2 + len(test_tracks) ** 2
		
		if n_samples == 0:
			train_tracks.extend([track.index for track in versions_sets[current_class[0]]])
		else:
			train_percentage = (len(train_tracks) ** 2) / n_samples
			test_percentage = (len(test_tracks) ** 2) / n_samples
			
			train_distance = train_test_split - train_percentage
			test_distance = (1 - train_test_split) - test_percentage
			
			if train_distance >= test_distance:
				train_tracks.extend([track.index for track in versions_sets[current_class[0]]])
			elif train_distance < test_distance:
				test_tracks.extend([track.index for track in versions_sets[current_class[0]]])
		
		count_tracks = count_tracks[1:]
	
	assert len(np.intersect1d(train_tracks, test_tracks)) == 0
	assert len(train_tracks) + len(test_tracks) == len(tracks)
		
	return train_tracks, test_tracks

def compute_filters(tracks, train_tracks, test_tracks):
	# train filter
	train_filter = np.zeros((len(tracks), len(tracks)), dtype=bool)
	
	train_product = np.array(list(product(train_tracks, train_tracks)))
	
	for index in train_product:
		i, j = index
		train_filter[i, j] = True
	
	# test filter
	test_filter = np.zeros((len(tracks), len(tracks)), dtype=bool)
	
	test_product = np.array(list(product(test_tracks, test_tracks)))
	
	for index in test_product:
		i, j = index
		test_filter[i, j] = True
	
	return train_filter, test_filter

def gather_samples(tensor, labels):
	samples_0s = []
	samples_1s = []
	
	n = len(labels)
	for i in range(n):
		if labels[i] == 0:
			samples_0s.append(tensor[i])
		elif labels[i] == 1:
			samples_1s.append(tensor[i])
		else:
			raise ValueError("invalid label")
	
	return np.array(samples_0s), np.array(samples_1s)

