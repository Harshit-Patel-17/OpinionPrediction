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
	m = m + 1

from random import shuffle
indices = range(m)
shuffle(indices)
X = X[indices]
Y = Y[indices]

#Find best parameters
from sklearn import svm, tree, cross_validation, naive_bayes, grid_search

criterions = ['gini', 'entropy']
max_depths = [None, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
min_samples_splits = [2, 3, 4, 5, 6, 7, 8, 9, 10]

'''
print "<table>"

print "<tr>"
print "<th>", "criterion", "</th>"
print "<th>", "max_depth", "</th>"
print "<th>", "min_samples_split", "</th>"
print "<th>", "Accuracy", "</th>"
print "</tr>"
'''

print "criterion , max_depth , min_samples_split , Accuracy"
for criterion in criterions:
	for max_depth in max_depths:
		for min_samples_split in min_samples_splits:
			clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split)
			scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
			print criterion, ',', max_depth, ',', min_samples_split, ',', np.average(scores)
			'''
			print "<tr>"
			print "<td>", criterion, "</td>"
			print "<td>", max_depth, "</td>"
			print "<td>", min_samples_split, "</td>"
			print "<td>", np.average(scores), "</td>"
			print "</tr>"
			'''

#print "</table>"