alphas='0 2 4 6 8 10'
algorithms='lp_zgl adsorption mad'
train_sizes='200 600 1000 1600 2000'
for alpha in $alphas; do
	for algorithm in $algorithms; do
		for train_size in $train_sizes; do
			echo "# File where the input graph is stored. Format of the data
# is specified by the "data_format" field. The most common
# format is the edge_factored way where each line specifies an
# edge in the graph. For example,
# <source_node>TAB<target_node>TAB<edge_weight>
#
graph_file = ../graphs/similarity_graph_$alpha.txt
data_format = edge_factored

# Specifies the seed label information to be injected into
# selected nodes at the start of the algorithm.
seed_file = ../seeds/train_nodes_$train_size.txt

# Gold labels of nodes (if known). This gold label information
# is used only during evaluation.
gold_labels_file = ../gold_nodes/gold_nodes_$train_size.txt

# Nodes (along with corresponding gold label information) which
# are used during evaluation, if any. This is kept as separate
# option from the gold_labels_file, as we may not evaluate all
# nodes whose gold label information is known. 
test_file = ../gold_nodes/gold_nodes_$train_size.txt

# Number of label propagation rounds
iters = 10
verbose = true

# All nodes with degree lower than this threshold will be pruned
# away. No nodes will be pruned with a threshold of 0
prune_threshold = 0

# Choose one of the three label propagations algorithms
#algo = lp_zgl
#algo = adsorption
#algo = mad
algo = $algorithm

# Hyperparameters for Adsorption and MAD
mu1 = 1
mu2 = 1e-2
mu3 = 1e-2
beta = 2

# File where label propagation output is stored. Each line
# corresponds to a node and fields in the line are organized
# as follows:
#
# <node_name>TAB[<gold_label> <gold_score>]+TAB[<seed_label> <seed_score>]+TAB\
#	[<estimated_label> <estimated_score>]+TAB<is_test_node>TAB<node_MRR>
#
# For nodes for which gold (or seed label) information is not known, the corresponding
# field is left empty.
#
output_file = ../output_nodes/label_prop_output_${train_size}_${alpha}_${algorithm}.txt
" > config
junto config config
		done
	done
done