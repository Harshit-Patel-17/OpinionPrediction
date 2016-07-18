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
data = pd.read_csv("../data/PoliticianData2Classes.csv", header=None)
politician_indices = pd.read_csv("../corpus/PoliticianLabels.txt", header=None)
topic_indices = pd.read_csv("../corpus/TopicLabels.txt", header=None)
opinion_indices = pd.read_csv("../corpus/OpinionLabels.txt", header=None)

#Build train and test nodes
m = data.shape[0]

train_size = [200, 600, 1000, 1600, 2000]

from random import shuffle
indices = range(m)
shuffle(indices)

for size in train_size:
	train_file = "../data/seeds/train_nodes_" + `size` + ".txt"
	test_file = "../data/gold_nodes/gold_nodes_" + `size` + ".txt"
	with open(train_file, "w") as train_nodes:
		with open(test_file, "w") as test_nodes:	
			for i in range(m):
				index = indices[i]
				row = data.loc[[index]]
				opinion = row[1][index].strip()
				line = "N" + `index` + "\t" + opinion + "\t1.0\n"
				if i < size:
					train_nodes.write(line)
				else:
					test_nodes.write(line)