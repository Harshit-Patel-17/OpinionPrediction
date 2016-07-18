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

documents = []
target = []
for index, row in data.iterrows():
	politician_index = politician_indices[politician_indices[0] == row[0].strip()].index.tolist()[0]
	opinion_index = opinion_indices[opinion_indices[0] == row[1].strip()].index.tolist()[0]
	topic_index = topic_indices[topic_indices[0] == row[2].strip()].index.tolist()[0]
	if(opinion_index == 2):
		continue
	documents.append(politician_corpora[politician_index].strip() + " " + topic_corpora[topic_index].strip())
	target.append(opinion_index)

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
			scores = cross_validation.cross_val_score(clf, X_tf_idf, target, cv=5)
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