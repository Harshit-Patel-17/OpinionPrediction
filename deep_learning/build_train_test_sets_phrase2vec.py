#Log various events
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#Build vectors
from gensim import corpora, models, similarities
documents = models.doc2vec.TaggedLineDocument("../corpus/PoliticiansCorpora.txt")
politicians_model = models.doc2vec.Doc2Vec(documents, size=300)

documents = models.doc2vec.TaggedLineDocument("../corpus/TopicsCorpora.txt")
topics_model = models.doc2vec.Doc2Vec(documents, size=300)

#Load Data
import numpy as np
import pandas as pd
data = pd.read_csv("../data/PoliticianData.csv", header=None)
politician_indices = pd.read_csv("../corpus/PoliticianLabels.txt", header=None)
topic_indices = pd.read_csv("../corpus/TopicLabels.txt", header=None)
opinion_indices = pd.read_csv("../corpus/OpinionLabels.txt", header=None)

#Build train data and test data
m = data.shape[0]
n = politicians_model.docvecs[0].shape[0] + topics_model.docvecs[0].shape[0]
X = np.zeros((m, n), dtype=np.float)
Y = np.zeros(m, dtype=np.int)

current_row = 0
for index, row in data.iterrows():
	politician_index = politician_indices[politician_indices[0] == row[0].strip()].index.tolist()[0]
	opinion_index = opinion_indices[opinion_indices[0] == row[1].strip()].index.tolist()[0]
	topic_index = topic_indices[topic_indices[0] == row[2].strip()].index.tolist()[0]
	politician_vec = politicians_model.docvecs[politician_index]
	topic_vec = topics_model.docvecs[topic_index]
	if(opinion_index == 2):
		continue
	X[current_row] = np.concatenate((politician_vec, topic_vec))
	Y[current_row] = opinion_index
	current_row = current_row + 1

#Shuffle data
from random import shuffle
indices = range(current_row)
shuffle(indices)
X = X[indices]
Y = Y[indices]

train_size = int(0.8 * current_row)

Xtrain = X[:train_size]
Ytrain = Y[:train_size]
Xtest = X[train_size:]
Ytest = Y[train_size:]

#Save the data
from six.moves import cPickle as pickle
pickle_file = 'data_phrase2vec.pickle'
try:
	f = open(pickle_file, 'wb')
	save = {
		'train_dataset': Xtrain,
		'train_labels': Ytrain,
		'test_dataset': Xtest,
		'test_labels': Ytest,
	}
	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	f.close()
except Exception as e:
	print('Unable to save data to ', pickle_file, ':', e)
	raise

import os
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)