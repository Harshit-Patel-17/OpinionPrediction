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

kernels = ['sigmoid']
Cs = [1, 10, 20, 30, 40, 50]
gammas = [0.0001, 0.001, 0.01, 0.1, 1]

'''
print "<table>"

print "<tr>"
print "<th>", "kernel", "</th>"
print "<th>", "C", "</th>"
print "<th>", "gamma", "</th>"
print "<th>", "Accuracy", "</th>"
print "</tr>"
'''

print "kernel , C , gamma , Accuracy\n"
with open("output3.txt", "a") as output:
		output.write("kernel , C , gamma , Accuracy\n")
for kernel in kernels:
	for C in Cs:
		for gamma in gammas:
			clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
			scores = cross_validation.cross_val_score(clf, X_tf_idf, target, cv=5)
			line = kernel + " , " + `C` + " , " + `gamma` + " , " + `np.average(scores)` + "\n"
			print line
			with open("output3.txt", "a") as output:
				output.write(line)
			'''
			print "<tr>"
			print "<td>", kernel, "</td>"
			print "<td>", C, "</td>"
			print "<td>", gamma, "</td>"
			print "<td>", np.average(scores), "</td>"
			print "</tr>"
			'''

#print "</table>"