#Load Data
import numpy as np
import pandas as pd
data = pd.read_csv("../data/PoliticianData2Classes.csv", header=None)
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

#Build n-grams
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(ngram_range=(3,3))
politician_counts = count_vect.fit_transform(politician_corpora)
topic_counts = count_vect.fit_transform(topic_corpora)

#Apply tf-idf downscaling
from sklearn.feature_extraction.text import TfidfTransformer
tf_idf_transformer = TfidfTransformer()
politician_tf_idf = tf_idf_transformer.fit_transform(politician_counts)
topic_tf_idf = tf_idf_transformer.fit_transform(topic_counts)

#Compute similarity
politician_similarity = []
topic_similarity = []
from sklearn.metrics.pairwise import linear_kernel
for i in range(politician_indices.shape[0]):
	politician_similarity.append(linear_kernel(politician_tf_idf[i], politician_tf_idf).flatten())
for i in range(topic_indices.shape[0]):
	topic_similarity.append(linear_kernel(topic_tf_idf[i], topic_tf_idf).flatten())

m = data.shape[0]
#Generate graph
print "Generating graph"
base = "../data/ngrams_graphs/graphs/"
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
								p_similarity = politician_similarity[P1_index][P2_index]
								t_similarity = topic_similarity[T1_index][T2_index]
								lines = []
								for alpha in range(0, 6):
									similarity = (0.2*alpha) * p_similarity + (1 - 0.2*alpha) * t_similarity
									lines.append("N" + `i` + "\t" + "N" + `j` + "\t" + `similarity` + "\n")
								similarity_graph_0.write(lines[0])
								similarity_graph_2.write(lines[1])
								similarity_graph_4.write(lines[2])
								similarity_graph_6.write(lines[3])
								similarity_graph_8.write(lines[4])
								similarity_graph_10.write(lines[5])
