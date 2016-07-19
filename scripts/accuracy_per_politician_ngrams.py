#Load Data
import numpy as np
import pandas as pd
data = pd.read_csv("../data/PoliticianData.csv", header=None)
politician_indices = pd.read_csv("../corpus/PoliticianLabels.txt", header=None)
topic_indices = pd.read_csv("../corpus/TopicLabels.txt", header=None)
opinion_indices = pd.read_csv("../corpus/OpinionLabels.txt", header=None)

#Read politician corpora
politician_corpora = []
with open("../corpus/PoliticiansCorpora.txt", "r") as infile:
	for line in infile:
		politician_corpora.append(line)

#Read topic corpora
topic_corpora = []
with open("../corpus/TopicsCorpora.txt", "r") as infile:
	for line in infile:
		topic_corpora.append(line)

politicians_classification = []
for index, row in politician_indices.iterrows():
	politicians_classification.append((row[0], []))

documents = []
target = []
m = 0
for index, row in data.iterrows():
	politician_index = politician_indices[politician_indices[0] == row[0].strip()].index.tolist()[0]
	opinion_index = opinion_indices[opinion_indices[0] == row[1].strip()].index.tolist()[0]
	topic_index = topic_indices[topic_indices[0] == row[2].strip()].index.tolist()[0]
	if(opinion_index == 2):
		continue
	documents.append(politician_corpora[politician_index].strip() + " " + topic_corpora[topic_index].strip())
	target.append(opinion_index)
	politicians_classification[politician_index][1].append(m)
	m = m + 1

target = np.asarray(target)

#Extract features
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(ngram_range=(3,3))
X_counts = count_vect.fit_transform(documents)

#Apply tf-idf downscaling
from sklearn.feature_extraction.text import TfidfTransformer
tf_idf_transformer = TfidfTransformer()
X_tf_idf = tf_idf_transformer.fit_transform(X_counts)

#Shuffle data
from random import shuffle
indices = range(len(target))
shuffle(indices)
X_tf_idf = X_tf_idf[indices]
target = target[indices]

reverse_mapping = np.zeros(m, dtype=np.int)
for i in range(m):
	reverse_mapping[indices[i]] = i

#Train model	
from sklearn import naive_bayes, tree, cross_validation, svm, linear_model, semi_supervised

clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=7)
dt_prediction = cross_validation.cross_val_predict(clf, X_tf_idf, target, cv=5)

clf = svm.SVC(kernel="sigmoid", C=10, gamma=1)
svm_prediction = cross_validation.cross_val_predict(clf, X_tf_idf, target, cv=5)

dt_prediction = dt_prediction[reverse_mapping]
svm_prediction = svm_prediction[reverse_mapping]
Y = target[reverse_mapping]

print "politician , decision_tree , svm"
for (name, indices) in politicians_classification:
	dt_pred = dt_prediction[indices]
	svm_pred = svm_prediction[indices]
	y = Y[indices]
	dt_accuracy = float(len(np.where(dt_pred == y)[0])) / len(y)
	svm_accuracy = float(len(np.where(svm_pred == y)[0])) / len(y)
	print name, ",", dt_accuracy, ",", svm_accuracy

