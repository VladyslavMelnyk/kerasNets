from keras.models import load_model
import cPickle as pickle
import time

from dataset import *

switch_to_cpu()

model_location = os.path.join("models", "seresnet50.hdf5")

out_dir = "./outdir"

if not os.path.exists(out_dir):
	os.mkdir(out_dir)

test_output = os.path.join(out_dir, "results_test.p")

model = load_model(model_location)

batch_size = 1

test = read_sets_file(imagesets_folder, "test")

input_shape = (1000, 600, 3)

out = model.evaluate_generator(
	DataGenerator(img_folder=image_folder, annot_folder=annotation_folder, filenames=test, classes=classes,
	              input_shape=input_shape, batch_size=batch_size, category_repr=True),
	steps=len(test)/batch_size
)

print "Loss: {0}".format(out)

th_step = 0.05
thresholds_array = np.arange(0, 1, th_step)

if not os.path.exists(test_output):
	j = 0
	avg_time_for_inference = 0
	tp = dict(zip(thresholds_array, [dict(zip(classes, [0] * len(classes))) for i in range(len(thresholds_array))]))
	tn = dict(zip(thresholds_array, [dict(zip(classes, [0] * len(classes))) for i in range(len(thresholds_array))]))
	fp = dict(zip(thresholds_array, [dict(zip(classes, [0] * len(classes))) for i in range(len(thresholds_array))]))
	fn = dict(zip(thresholds_array, [dict(zip(classes, [0] * len(classes))) for i in range(len(thresholds_array))]))

	precision_per_label = dict(zip(thresholds_array, [dict(zip(classes, [0] * len(classes)))
	                                                  for i in range(len(thresholds_array))]))
	recall_per_label = dict(zip(thresholds_array, [dict(zip(classes, [0] * len(classes)))
	                                               for i in range(len(thresholds_array))]))
	f1_score_per_label = dict(zip(thresholds_array, [dict(zip(classes, [0] * len(classes)))
	                                                 for i in range(len(thresholds_array))]))
	accuracy_per_label = dict(zip(thresholds_array, [dict(zip(classes, [0] * len(classes)))
	                                                 for i in range(len(thresholds_array))]))

	avg_precision = dict(zip(thresholds_array, [0] * len(thresholds_array)))
	avg_recall = dict(zip(thresholds_array, [0] * len(thresholds_array)))
	avg_f1 = dict(zip(thresholds_array, [0] * len(thresholds_array)))
	avg_acc = dict(zip(thresholds_array, [0] * len(thresholds_array)))

	img_generator = DataGenerator(img_folder=image_folder, annot_folder=annotation_folder, filenames=test,
	                              classes=classes, input_shape=input_shape, batch_size=batch_size, category_repr=False)

	while True:
		try:
			x_test, ground_truth = img_generator.next()
			if j >= len(test):
				break
		except StopIteration:
			break

		start_time = time.time()
		predictions = model.predict(x_test)
		j += 1
		elapsed_time = time.time() - start_time
		print "Seconds per frame # {0}: {1}".format(i, elapsed_time)
		avg_time_for_inference += elapsed_time

		for it, single_prediction in enumerate(predictions):
			for th in thresholds_array:
				mask = np.where(single_prediction > th)[0]
				pr = set(np.array(classes)[mask])

				tp_set = ground_truth[it].intersection(pr)
				update_metric(tp, th, tp_set)

				fn_set = ground_truth[it] - tp_set
				update_metric(fn, th, fn_set)

				fp_set = pr - tp_set
				update_metric(fp, th, fp_set)

				tn_set = set(classes) - pr - ground_truth[it]
				update_metric(tn, th, tn_set)

	for th in thresholds_array:
		for cls in classes:
			precision_per_label[th][cls] = precision(tp[th][cls], fp[th][cls])
			recall_per_label[th][cls] = recall(tp[th][cls], fn[th][cls])
			f1_score_per_label[th][cls] = f1_score(precision_per_label[th][cls], recall_per_label[th][cls])
			accuracy_per_label[th][cls] = accuracy(tp[th][cls], tn[th][cls], fp[th][cls], fn[th][cls])

			avg_precision[th] += precision_per_label[th][cls] / len(classes)
			avg_recall[th] += recall_per_label[th][cls] / len(classes)
			avg_f1[th] += f1_score_per_label[th][cls] / len(classes)
			avg_acc[th] += accuracy_per_label[th][cls] / len(classes)

	# dump all test results to file
	pickle.dump([precision_per_label, recall_per_label, f1_score_per_label, accuracy_per_label, avg_precision,
	             avg_recall, avg_f1, avg_acc], open(test_output, "wb"))
	avg_time_for_inference /= (len(test) / batch_size)
	print "Average inference time for one image: {0}".format(avg_time_for_inference)
else:
	# read all test result from file
	precision_per_label, recall_per_label, f1_score_per_label, accuracy_per_label, avg_precision, \
		avg_recall, avg_f1, avg_acc = pickle.load(open(test_output, "rb"))

print_table("Precision", thresholds_array, precision_per_label, avg_precision, classes)
print_table("Recall", thresholds_array, recall_per_label, avg_recall, classes)
print_table("F1 Score", thresholds_array, f1_score_per_label, avg_f1, classes)
print_table("Accuracy", thresholds_array, accuracy_per_label, avg_acc, classes)

best_match = {}

for cls in classes:
	max_value = -1
	for th in thresholds_array:
		if accuracy_per_label[th][cls] > max_value:
			max_value = accuracy_per_label[th][cls]
			best_match[cls] = [max_value, th]

avg_p = 0
for cls in classes:
	avg_p += best_match[cls][0]

print "*" * 180
print "{:>15}\t{}".format("Class", "".join('{:<9}\t'.format(k) for k in classes))
print "{:>15}\t{}".format("Threshold", "".join('{:<9}\t'.format(best_match[k][1]) for k in classes))
print "{:>15}\t{}".format("Value", "".join('{:<9.5f}\t'.format(best_match[k][0]) for k in classes))
print '*' * 180
print "{:>15}\t{}".format("Max accuracy", avg_p / len(classes))
print '*' * 180
k, v = dict_max_value(avg_acc)
print "Accuracy with global threshold: {0:.3} - {1:.3}".format(k, v)
k, v = dict_max_value(avg_f1)
print "F1 with threshold: {0:.3} - {1:.3}".format(k, v)
k, v = dict_max_value(avg_recall)
print "Recall with threshold: {0:.3} - {1:.3}".format(k, v)
k, v = dict_max_value(avg_precision)
print "Precision with threshold: {0:.3} - {1:.3}".format(k, v)



