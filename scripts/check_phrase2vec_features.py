#Check distinguishing power of phrase2vec features

from __future__ import print_function
import itertools
import numpy as np
import pandas as pd
import logging
from gensim import corpora, models, similarities
import matplotlib.pyplot as plt

#Generate similarity mean and srandard deviation
def generate_similarity(list1, list2, model):
	similarity_scores = [model.docvecs.similarity(index1, index2)
							for (index1, index2) in list(itertools.product(list1, list2))]
	return (np.mean(similarity_scores), np.std(similarity_scores))

#Log various events
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#Load vectors
politicians_model = models.doc2vec.Doc2Vec.load("../saved_models/PoliticianCorporaVectors.txt")
topics_model = models.doc2vec.Doc2Vec.load("../saved_models/TopicsCorporaVectors.txt")

#Load Data
data = pd.read_csv("../data/PoliticianData.csv", header=None)
politician_indices = pd.read_csv("../corpus/PoliticianLabels.txt", header=None)
topic_indices = pd.read_csv("../corpus/TopicLabels.txt", header=None)
opinion_indices = pd.read_csv("../corpus/OpinionLabels.txt", header=None)

topic_wise_favor = [[] for i in range(topic_indices.shape[0])] #Indices of politicians favoring a topic
topic_wise_oppose = [[] for i in range(topic_indices.shape[0])] #Indices of politicians opposing a topic
politician_wise_favor = [[] for i in range(politician_indices.shape[0])] #Indices of topics supported by a politician
politician_wise_oppose = [[] for i in range(politician_indices.shape[0])] #Indices of topics opposed by a politician

#Construct lists
for index, row in data.iterrows():
	politician_index = politician_indices[politician_indices[0] == row[0].strip()].index.tolist()[0]
	opinion_index = opinion_indices[opinion_indices[0] == row[1].strip()].index.tolist()[0]
	topic_index = topic_indices[topic_indices[0] == row[2].strip()].index.tolist()[0]
	if(opinion_index == 0):
		topic_wise_favor[topic_index].append(politician_index)
		politician_wise_favor[politician_index].append(topic_index)
	elif(opinion_index == 1):
		topic_wise_oppose[topic_index].append(politician_index)
		politician_wise_oppose[politician_index].append(topic_index)
	elif(opinion_index == 2):
		continue

topicFF_mean = []
topicFF_std = []
topicFO_mean = []
topicFO_std = []
topicOO_mean = []
topicOO_std = []
topicLabels = []
for i in range(topic_indices.shape[0]):
	topicLabels.append('T' + `i`)
	similarity = generate_similarity(topic_wise_favor[i], topic_wise_favor[i], politicians_model)
	topicFF_mean.append(similarity[0])
	topicFF_std.append(similarity[1])
	print("Topic", i, ",", "favor-favor", ",", similarity[0], ",", similarity[1])
	similarity = generate_similarity(topic_wise_favor[i], topic_wise_oppose[i], politicians_model)
	topicFO_mean.append(similarity[0])
	topicFO_std.append(similarity[1])
	print("Topic", i, ",", "favor-oppose", ",", similarity[0], ",", similarity[1])
	similarity = generate_similarity(topic_wise_oppose[i], topic_wise_oppose[i], politicians_model)
	topicOO_mean.append(similarity[0])
	topicOO_std.append(similarity[1])
	print("Topic", i, ",", "oppose-oppose", ",", similarity[0], ",", similarity[1])

politicianFF_mean = []
politicianFF_std = []
politicianFO_mean = []
politicianFO_std = []
politicianOO_mean = []
politicianOO_std = []
politicianLabels = []
for i in range(politician_indices.shape[0]):
	politicianLabels.append('P' + `i`)
	similarity = generate_similarity(politician_wise_favor[i], politician_wise_favor[i], topics_model)
	politicianFF_mean.append(similarity[0])
	politicianFF_std.append(similarity[1])
	print("Politician", i, ",", "favor-favor", ",", similarity[0], ",", similarity[1])
	similarity = generate_similarity(politician_wise_favor[i], politician_wise_oppose[i], topics_model)
	politicianFO_mean.append(similarity[0])
	politicianFO_std.append(similarity[1])
	print("Politician", i, ",", "favor-oppose", ",", similarity[0], ",", similarity[1])
	similarity = generate_similarity(politician_wise_oppose[i], politician_wise_oppose[i], topics_model)
	politicianOO_mean.append(similarity[0])
	politicianOO_std.append(similarity[1])
	print("Politician", i, ",", "oppose-oppose", ",", similarity[0], ",", similarity[1])

def autolabel(rects, ax):
	for rect in rects:
		height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%f' % height, ha='center', va='bottom')

def draw_bar_chart(means1, means2, means3, std1, std2, std3, labels, title, legend, xlabel, ylabel):
	N = len(means1)
	ind = np.arange(N)
	width = 0.30

	fig, ax = plt.subplots()

	rects1 = ax.bar(ind, means1, width, color="MediumSlateBlue")
	rects2 = ax.bar(ind + width, means2, width, color="Tomato")
	rects3 = ax.bar(ind + 2*width, means3, width, color="Green")

	axes = plt.gca()
	#axes.set_ylim([0, 1])

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.set_xticks(ind + 1.5*width)
	ax.set_xticklabels(labels)

	ax.legend((rects1[0], rects2[0], rects3[0]), legend)

	#autolabel(rects1, ax)
	#autolabel(rects2, ax)
	#autolabel(rects3, ax)

	plt.show()

draw_bar_chart(topicFF_mean, topicFO_mean, topicOO_mean, topicFF_std, topicFO_std, topicOO_std, topicLabels, 
	"Average similarity between politicians for a topic using phrase2vec", 
	("Both politicians Favor the topic", "Both politicians have different opinions on the topic", "Both politicians Oppose the topic"),
	"Topics", "Average Similarity of politicians")
draw_bar_chart(politicianFF_mean[:20], politicianFO_mean[:20], politicianOO_mean[:20], politicianFF_std[:20], politicianFO_std[:20], politicianOO_std[:20], politicianLabels[:20], 
	"Average similarity between topics for a politician using phrase2vec",
	("Politician Favors both the topics", "Politician has different opinions on topics", "Politician Opposes both the topics"),
	"Politicians", "Average Similarity of topics")
