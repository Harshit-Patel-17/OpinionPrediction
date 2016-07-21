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

train_size = int(0.8 * len(target))

Xtrain = X_tf_idf[:train_size]
Ytrain = target[:train_size]
Xtest = X_tf_idf[train_size:]
Ytest = target[train_size:]

#Save the data
from six.moves import cPickle as pickle
pickle_file = 'data_ngrams.pickle'
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