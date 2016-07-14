import numpy as np
import pandas as pd

threshold = 0.0

#similarity_graph_0
for i in range(6):
	src_graph_file = '../data/similarity_graph_' + `2*i` + '.txt'
	dst_graph_file = '../data/similarity_graph_' + `2*i` + '_filtered.txt'
	graph = pd.read_csv(src_graph_file, sep='\t', header=None)
	filtered_graph = graph[graph[2] > threshold]
	filtered_graph.to_csv(dst_graph_file, sep='\t', header=None, index=None)