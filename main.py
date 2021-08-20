import sys

import dataset as ds
import samples as s
import classification as c

def main(dataset, n_models, train_test_split, classifier_name):
	print("this dataset has:")
	
	classes, tracks = ds.load(dataset)
	print(len(classes), "classes")
	print(len(tracks), "tracks")
	print()
	
	dataset_name = dataset.split('/')[-1]
	csv_filename = dataset_name + ".csv"
	
	labels, tensor = s.get_tensor(csv_filename, n_models, classes, tracks, dataset)
	print("using Top", n_models, "models a", tensor.shape, "tensor is built")
	print()
	
	train_filter, test_filter = s.get_filters(classes, tracks, train_test_split)
	train_samples, train_labels, test_samples, test_labels, test_samples_0s, test_samples_1s = s.get_samples_and_labels(labels, tensor, train_filter, test_filter)
	
	n_train = len(train_samples)
	n_test = len(test_samples)
	total = n_train + n_test
	print("after the train/test split process we have:")
	print(n_train, "train samples", "({}%)".format(round(100 * n_train / total, 3)))
	print(n_test, "test samples", "({}%)".format(round(100 * n_test / total, 3)))
	print()
	
	print("training", classifier_name, "...")
	classifier = getattr(c, classifier_name)(train_samples, train_labels)
	print()
	
	print("predicting labels...")
	predicted_labels = classifier.predict(test_samples)
	predicted_labels_0s = classifier.predict(test_samples_0s)
	predicted_labels_1s = classifier.predict(test_samples_1s)
	print()
	
	accuracy, accuracy_0s, accuracy_1s, precision, recall, f1_score = c.calculate_metrics(test_labels, predicted_labels, test_samples_0s, predicted_labels_0s, test_samples_1s, predicted_labels_1s)
	print("[classification metrics]")
	print("accuracy:", accuracy)
	print("accuracy_0s:", accuracy_0s)
	print("accuracy_1s:", accuracy_1s)
	print("precision:", precision)
	print("recall:", recall)
	print("f1_score:", f1_score)
	print()

if __name__ == "__main__":
	if len(sys.argv) != 5:
		print("error: invalid number of args")
		print()
		
		print("usage: python", sys.argv[0], "[dataset path] [number of models] [train/test split percentage] [classifier name]")
		print()
		
		sys.exit(1)
	
	dataset_param = sys.argv[1]
	n_models_param = int(sys.argv[2])
	train_test_split_param = float(sys.argv[3])
	classifier_name_param = sys.argv[4]
	main(dataset_param, n_models_param, train_test_split_param, classifier_name_param)

