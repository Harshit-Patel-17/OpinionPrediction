from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from random import shuffle

def process_docs(docs):
	texts = []
	
	for doc in docs:

		#Cleaning and tokenizing documents
		raw = doc.lower()
		tokens = tokenizer.tokenize(raw)

		#Remove stop-words from tokens
		stopped_tokens = [token for token in tokens if not token in en_stop]

		#Stem tokens
		stemmed_tokens = [p_stemmer.stem(token) for token in stopped_tokens]

		#Add tokens to list
		texts.append(stemmed_tokens)

	return texts


tokenizer = RegexpTokenizer(r'\w+')

#Create English stop-words list
en_stop = get_stop_words('en')

#Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

#Create documents
politician_docs = []
topic_docs = []

with open("../corpus/PoliticiansCorpora.txt", "r") as infile:
	for line in infile:
		politician_docs.append(line)

with open("../corpus/TopicsCorpora.txt", "r") as infile:
	for line in infile:
		topic_docs.append(line)

#List of tokenized documents
politician_texts = process_docs(politician_docs)
topic_texts = process_docs(topic_docs)

#Load dictionaries and models
politician_dictionary = corpora.dictionary.Dictionary.load("../saved_models/PoliticianDictionary")
topic_dictionary = corpora.dictionary.Dictionary.load("../saved_models/TopicDictionary")
politician_ldamodel = models.ldamodel.LdaModel.load("../saved_models/PoliticianLdaModel")
topic_ldamodel = models.ldamodel.LdaModel.load("../saved_models/TopicLdaModel")

politician_features = []
for text in politician_texts:
	politician_features.append(politician_ldamodel[politician_dictionary.doc2bow(text)])

topic_features = []
for text in topic_texts:
	topic_features.append(topic_ldamodel[topic_dictionary.doc2bow(text)])

#Load Data
import numpy as np
import pandas as pd
data = pd.read_csv("../data/PoliticianData.csv", header=None)
politician_indices = pd.read_csv("../corpus/PoliticianLabels.txt", header=None)
topic_indices = pd.read_csv("../corpus/TopicLabels.txt", header=None)
opinion_indices = pd.read_csv("../corpus/OpinionLabels.txt", header=None)

m = data.shape[0]
n = 200
X = np.zeros((m, n), dtype=np.float)
Y = np.zeros(m, dtype=np.int)

m = 0
for index, row in data.iterrows():
	politician_index = politician_indices[politician_indices[0] == row[0].strip()].index.tolist()[0]
	opinion_index = opinion_indices[opinion_indices[0] == row[1].strip()].index.tolist()[0]
	topic_index = topic_indices[topic_indices[0] == row[2].strip()].index.tolist()[0]
	if(opinion_index == 2):
		continue
	for (topic, fraction) in politician_features[politician_index]:
		X[m][topic] = fraction
	for (topic, fraction) in topic_features[topic_index]:
		X[m][topic + 100] = fraction
	Y[m] = opinion_index
	m = m + 1

#Shuffle data
indices = range(m)
shuffle(indices)
X = X[indices]
Y = Y[indices]

#Train model
from sklearn import svm, naive_bayes, linear_model, preprocessing, tree, cross_validation

clf = tree.DecisionTreeClassifier()
scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
print scores

clf = naive_bayes.MultinomialNB()
scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
print scores
