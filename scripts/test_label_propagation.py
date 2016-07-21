alphas = [0, 2, 4, 6, 8, 10]
algorithms = ['lp_zgl', 'adsorption', 'mad']
train_sizes = [200, 600, 1000, 1600, 2000]
accuracies = []

base = "../data/ngrams_graphs/output_nodes/"

print "algorithm , training_set_size , alpha , accuracy"

for alpha in alphas:
	for algorithm in algorithms:
		for train_size in train_sizes:
			correct_predictions = 0
			total_predictions = 0
			file_name = base + "label_prop_output_" + `train_size` + "_" + `alpha` + "_" + algorithm + ".txt"
			with open(file_name) as label_prop_output:
				for line in label_prop_output:
					row = line.split("\t")
					if(row[4] == 'true'):
						estimated_label = row[3].split(" ")[0]
						gold_label = row[1].split(" ")[0]
						if(estimated_label == gold_label):
							correct_predictions += 1
						total_predictions += 1
			accuracy = float(correct_predictions) / total_predictions
			print algorithm, ",", train_size, ",", alpha/10.0, ",", accuracy