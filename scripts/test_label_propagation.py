train_sizes = [200, 600, 1000, 1600, 2000]
accuracies = []

print "training_set_size , accuracy"

for size in train_sizes:
	correct_predictions = 0
	total_predictions = 0
	file_name = "../data/output_nodes/label_prop_output_" + `size` + ".txt"
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
	print size, ",", accuracy
	accuracies.append(accuracy)

#Plot learning curve
import matplotlib.pyplot as plt

val_acc, = plt.plot(train_sizes, accuracies, marker="o", linestyle="-")

plt.xlabel("Training set size")
plt.ylabel("Accuracy(%)")
plt.title("Accuracy in Label Propagation")
plt.legend([val_acc], ["On test data"], loc=3)

for i, j in zip(train_sizes, accuracies):
	plt.annotate(str(j)[:6], xy=(i-100, j+0.0001))


plt.show()