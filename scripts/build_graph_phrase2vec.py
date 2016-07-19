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
m = data.shape[0]

base = "../data/phrase2vec_graphs/graphs/"

#Build complete graph
with open(base + "similarity_graph_0.txt", "w") as similarity_graph_0:
	with open(base + "similarity_graph_2.txt", "w") as similarity_graph_2:
		with open(base + "similarity_graph_4.txt", "w") as similarity_graph_4:
			with open(base + "similarity_graph_6.txt", "w") as similarity_graph_6:
				with open(base + "similarity_graph_8.txt", "w") as similarity_graph_8:
					with open(base + "similarity_graph_10.txt", "w") as similarity_graph_10:
						for i in range(0, m-1):
							print i
							row1 = data.loc[[i]]
							P1_index = politician_indices[politician_indices[0] == row1[0][i].strip()].index.tolist()[0]
							T1_index = topic_indices[topic_indices[0] == row1[2][i].strip()].index.tolist()[0]
							for j in range(i+1, m):
								row2 = data.loc[[j]]
								P2_index = politician_indices[politician_indices[0] == row2[0][j].strip()].index.tolist()[0]
								T2_index = topic_indices[topic_indices[0] == row2[2][j].strip()].index.tolist()[0]
								politician_similarity = politicians_model.docvecs.similarity(P1_index, P2_index)
								topic_similarity = topics_model.docvecs.similarity(T1_index, T2_index)
								lines = []
								for alpha in range(0, 6):
									similarity = (0.2*alpha) * politician_similarity + (1 - 0.2*alpha) * topic_similarity
									lines.append("N" + `i` + "\t" + "N" + `j` + "\t" + `similarity` + "\n")
								similarity_graph_0.write(lines[0])
								similarity_graph_2.write(lines[1])
								similarity_graph_4.write(lines[2])
								similarity_graph_6.write(lines[3])
								similarity_graph_8.write(lines[4])
								similarity_graph_10.write(lines[5])


	