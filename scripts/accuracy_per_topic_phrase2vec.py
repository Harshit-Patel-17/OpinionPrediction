#Log various events
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#Load vectors
from gensim import corpora, models, similarities
politicians_model = models.doc2vec.Doc2Vec.load("../saved_models/PoliticianCorporaVectors.txt")
topics_model = models.doc2vec.Doc2Vec.load("../saved_models/TopicsCorporaVectors.txt")

#Load Data
import numpy as np
import pandas as pd
data = pd.read_csv("../data/PoliticianData.csv", header=None)
politician_indices = pd.read_csv("../corpus/PoliticianLabels.txt", header=None)
topic_indices = pd.read_csv("../corpus/TopicLabels.txt", header=None)
opinion_indices = pd.read_csv("../corpus/OpinionLabels.txt", header=None)

#Build train data and test data
m = data.shape[0]
#n = politicians_model.docvecs[0].shape[0] + topics_model.docvecs[0].shape[0]
n = politicians_model.docvecs[0].shape[0] + topics_model.docvecs[0].shape[0]
X = np.zeros((m, n), dtype=np.float)
Y = np.zeros(m, dtype=np.int)

topics_classification = []
for index, row in topic_indices.iterrows():
	topics_classification.append((row[0], []))

m = 0
for index, row in data.iterrows():
	politician_index = politician_indices[politician_indices[0] == row[0].strip()].index.tolist()[0]
	opinion_index = opinion_indices[opinion_indices[0] == row[1].strip()].index.tolist()[0]
	topic_index = topic_indices[topic_indices[0] == row[2].strip()].index.tolist()[0]
	politician_vec = politicians_model.docvecs[politician_index]
	topic_vec = topics_model.docvecs[topic_index]
	if(opinion_index == 2):
		continue
	X[m] = np.concatenate((politician_vec, topic_vec))
	Y[m] = opinion_index
	topics_classification[topic_index][1].append(m)
	m = m + 1

from random import shuffle
indices = range(m)
shuffle(indices)
X = X[indices]
Y = Y[indices]

reverse_mapping = np.zeros(m, dtype=np.int)
for i in range(m):
	reverse_mapping[indices[i]] = i

from sklearn import svm, tree, cross_validation, naive_bayes, grid_search

clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=3)
dt_prediction = cross_validation.cross_val_predict(clf, X, Y, cv=5)

clf = svm.SVC(kernel="rbf", C=30, gamma=0.1)
svm_prediction = cross_validation.cross_val_predict(clf, X, Y, cv=5)

dt_prediction = dt_prediction[reverse_mapping]
svm_prediction = svm_prediction[reverse_mapping]
Y = Y[reverse_mapping]

print "topic , decision_tree , svm"
for (name, indices) in topics_classification:
	dt_pred = dt_prediction[indices]
	svm_pred = svm_prediction[indices]
	y = Y[indices]
	dt_accuracy = float(len(np.where(dt_pred == y)[0])) / len(y)
	svm_accuracy = float(len(np.where(svm_pred == y)[0])) / len(y)
	print name, ",", dt_accuracy, ",", svm_accuracy
