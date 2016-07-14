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
'''
#parameters = {'criterion':['gini', 'entropy'], 'max_depth':[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
parameters = {'criterion':['gini', 'entropy'], 'max_depth':[25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]}
clf = tree.DecisionTreeClassifier()

gs = grid_search.GridSearchCV(clf, parameters)
gs.fit(X, Y)
print gs.best_score_
print gs.best_estimator_.criterion
print gs.best_estimator_.max_depth
'''

#Train
from sklearn.learning_curve import learning_curve
clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=35)
train_sizes, train_scores, valid_scores = learning_curve(clf, X, Y, cv=10)
print train_sizes
print train_scores
print np.average(valid_scores, axis=1)
#scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
#print scores

#Plot learning curve
import matplotlib.pyplot as plt

trn_acc, = plt.plot(train_sizes, np.average(train_scores, axis=1), marker="o", linestyle="-")
val_acc, = plt.plot(train_sizes, np.average(valid_scores, axis=1), marker="o", linestyle="--")

plt.xlabel("Training set size")
plt.ylabel("Accuracy(%)")
plt.title("Accuracy in Decision Tree")
plt.legend([trn_acc, val_acc], ["On training data", "On test data"], loc=4)

for i, j in zip(train_sizes, np.average(train_scores, axis=1)):
	plt.annotate(str(j)[:4], xy=(i-100, j+0.01))

for i, j in zip(train_sizes, np.average(valid_scores, axis=1)):
	plt.annotate(str(j)[:4], xy=(i-100, j+0.01))

plt.show()