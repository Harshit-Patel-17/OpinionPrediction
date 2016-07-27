#For each politician predict the most frequent opinion

from __future__ import print_function

#Load Data
import numpy as np
import pandas as pd
data = pd.read_csv("../data/PoliticianData.csv", header=None)
politician_indices = pd.read_csv("../corpus/PoliticianLabels.txt", header=None)
topic_indices = pd.read_csv("../corpus/TopicLabels.txt", header=None)
opinion_indices = pd.read_csv("../corpus/OpinionLabels.txt", header=None)

politician_opinions = [[]] * politician_indices.shape[0] #Will contain list of opinions for each politician

#Construct list of opinions
for index, row in data.iterrows():
	politician_index = politician_indices[politician_indices[0] == row[0].strip()].index.tolist()[0]
	opinion_index = opinion_indices[opinion_indices[0] == row[1].strip()].index.tolist()[0]
	topic_index = topic_indices[topic_indices[0] == row[2].strip()].index.tolist()[0]
	if(opinion_index == 2):
		continue
	politician_opinions[politician_index].append(opinion_index)

#Predict the most frequent opinion for each politician
from scipy import stats
correct_predictions = 0
total_predictions = 0
for opinions in politician_opinions:
	most_frequent_opinion = stats.mode(opinions)[0][0]
	correct_predictions += stats.mode(opinions)[1][0]
	total_predictions += len(opinions)

print("Accuracy is", 100.0 * correct_predictions / total_predictions)