from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def naive_bayes(train_x, train_y):
	clf = GaussianNB()
	clf.fit(train_x, train_y)
	return clf

def logistic_regression(train_x, train_y):
	clf = LogisticRegression()
	clf.fit(train_x, train_y)
	return clf

def decision_tree(train_x, train_y):
	clf = DecisionTreeClassifier()
	clf.fit(train_x, train_y)
	return clf

def random_forest(train_x, train_y):
	clf = RandomForestClassifier()
	clf.fit(train_x, train_y)
	return clf

def ada_boost(train_x, train_y):
	clf = AdaBoostClassifier()
	clf.fit(train_x, train_y)
	return clf

def knn(train_x, train_y):
	clf = KNeighborsClassifier()
	clf.fit(train_x, train_y)
	return clf

def svm(train_x, train_y):
	clf = SVC()
	clf.fit(train_x, train_y)
	return clf

def calculate_metrics(test_labels, predicted_labels, test_samples_0s, predicted_labels_0s, test_samples_1s, predicted_labels_1s):
	accuracy = accuracy_score(test_labels, predicted_labels)
	accuracy_0s = accuracy_score(len(test_samples_0s) * [0], predicted_labels_0s)
	accuracy_1s = accuracy_score(len(test_samples_1s) * [1], predicted_labels_1s)
	precision = precision_score(test_labels, predicted_labels)
	recall = recall_score(test_labels, predicted_labels)
	f1 = f1_score(test_labels, predicted_labels)
	return accuracy, accuracy_0s, accuracy_1s, precision, recall, f1

