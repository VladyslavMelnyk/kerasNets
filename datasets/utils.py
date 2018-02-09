from __future__ import division

import os
import time
import cPickle as pickle

import cv2
import tensorflow as tf
import numpy as np
from xml.dom import minidom as xml

from keras.preprocessing.image import Iterator
import keras.backend.tensorflow_backend as K
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import rescale


def set_fraction_of_gpu_memory(gpu_fraction=0.3):
    """
	This function set up fraction of memory that will be used in training
	:param gpu_fraction:
	:return:
	"""
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(session)


def switch_to_cpu():
    """
	Force to use cpu instead of gpu
	:return: Nothing
	"""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def parent_directory(file_path, level):
    """
	Return parent directory of the file with respect to hierarchical level
	:param file_path: abs path to file
	:param level: level of hierarchy
	:return: path to parent directory
	"""
    if level == 0:
        return os.path.dirname(file_path)
    else:
        return os.path.dirname(parent_directory(file_path, level - 1))


def process_image(image_file, input_shape):
    """
	Read and reshape image
	:param image_file: path to image
	:param input_shape: new size
	:return: resized image
	"""
    im = cv2.imread(image_file, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    im_resize = cv2.resize(im, dsize=(input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)
    # normalized = (im_resize - img_mean) / img_std
    # return normalized
    return im_resize


def preprocess_fc(img, input_shape, rescale=1. / 255):
    img = cv2.imread(img, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out_img = np.zeros(img.shape)
    out_img[:, :, 0] = img_gray
    out_img[:, :, 1] = img_gray
    out_img[:, :, 2] = img_gray
    out_img = cv2.resize(out_img, (input_shape[1], input_shape[0]))
    out_img = out_img * rescale
    return out_img


def process_annotation(annot_file):
    """
	Extract labels of objects from annotation xml file
	:param annot_file: path to file
	:return: labels
	"""
    xmldoc = xml.parse(annot_file)
    objects = []
    for obj in xmldoc.getElementsByTagName("object"):
        deleted = obj.getElementsByTagName("deleted")
        if deleted and int(deleted[0].firstChild.nodeValue) == 1:
            continue
        else:
            objects.append(obj.getElementsByTagName("name")[0].firstChild.nodeValue)

    return objects


def to_categorical(y, labels):
    """
	Convert labels to binary vectors in order to pass them to network
	:param y: input vector
	:param labels: full list of labels or classes presented in dataset
	:return: categorical representation
	"""
    m = len(labels)
    categorical = np.zeros((1, m))

    for item in y:
        idx = labels.index(item)
        categorical[0, idx] = 1

    return categorical


def read_sets_file(folder, filename):
    """
	Read dataset's items from file
	:param folder: Folder that contains dataset's description files
	:param filename: name of file with set
	:return: list of files
	"""
    set_file = os.path.join(folder, filename + ".txt")

    if not os.path.exists(set_file):
        raise Exception("No file: {0}".format(set_file))

    with open(set_file, "r") as f:
        lines = f.read()

    return lines.split("\n")[:-1]


def update_metric(metric, threshold, labels):
    """
	Update given metric with respect to given threshold and label
	:param metric: Metric
	:param threshold: Threshold
	:param labels: labels or classes
	:return:
	"""
    for label in labels:
        metric[threshold][label] += 1


def precision(true_positive, false_positive):
    """
	Precision metric
	"""
    return true_positive / np.maximum(true_positive + false_positive, np.finfo(np.float64).eps)


def recall(true_positive, false_negative):
    """
	Recall metric
	"""
    return true_positive / np.maximum(true_positive + false_negative, np.finfo(np.float64).eps)


def f1_score(precision, recall):
    """
	F1 score metric
	"""
    return 2 * precision * recall / np.maximum(precision + recall, np.finfo(np.float64).eps)


def accuracy(true_positive, true_negative, false_positive, false_negative):
    """
	Accuracy metric
	"""
    return (true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive)


def print_table(table_name, thresholds_arr, value_arr, avg_value_arr, labels):
    """
	Print to console information about metrics per threshold per label
	:param table_name: Name of the metric
	:param thresholds_arr: Array of available thresholds
	:param value_arr: Array of respected values of metric
	:param avg_value_arr: array of respected average value of metric
	:param labels: all presented labels in dataset
	:return: Nothing
	"""
    info_str = ''
    info_str += '{}\n{}\n{}\n{}\n{}\n'.format(
        "*" * 180,
        "{:^180}".format(table_name),
        "*" * 180,
        "{:<15}\t{}".format("Class\Threshold", "".join('{:<6.5f}\t'.format(k) for k in thresholds_arr)),
        "*" * 180
    )
    for cls in labels:
        info_str += '{:>15}\t'.format(cls)
        for th in thresholds_arr:
            info_str += "{:<6.5f}\t".format(value_arr[th][cls])
        info_str += "\n"

    info_str += "*" * 180 + "\n"
    info_str += '{:>15}\t'.format("Avg " + table_name)
    for th in thresholds_arr:
        info_str += '{:<6.5f}\t'.format(avg_value_arr[th])
    info_str += "\n"

    print info_str


def dict_max_value(dct):
    """
	Find max value and respect key in dictionary
	:param dct: input dictionary
	:return: respect key and value
	"""
    key = dct.keys()[dct.values().index(max(dct.values()))]
    return key, dct[key]


def voting(model, x_input):
    vote = []
    for x in x_input:
        temp = np.array([x, cv2.flip(x, 1), cv2.flip(x, 0), cv2.flip(x, -1)])
        predictions = model.predict(temp)
        avg = np.average(predictions, axis=0)
        var = np.var(predictions, axis=0)
        vote.append(avg)
    vote = np.array(vote)
    return vote, predictions, var


def voting_ensemble(models, x_input):
    vt = []
    for model in models:
        vote, pr, vr = voting(model, x_input)
        vt.append(vote)

    vt = np.array(vt)
    return np.average(vt, axis=0)


def model_evaluation(model, data_generator, thresholds_array, classes, test_files, batch_size, out_pickle, out_pred):
    if not os.path.exists(out_pickle):
        f = open(out_pred, "w+")
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

        for i, file_name in enumerate(test_files):
            f.write("{0}\n".format(file_name))
            try:
                x_test, ground_truth = data_generator.next()
            except StopIteration:
                break
            # ground_truth = set(np.array(classes)[ground_truth.flatten().astype(np.bool)])
            start_time = time.time()
            # predictions = model.predict(x_test)
            # introduce voting
            predictions, prd, var = voting(model, x_test)
            # predictions = voting_ensemble(model, x_test)
            elapsed_time = time.time() - start_time
            # print "Seconds per frame # {0}: {1}".format(i, elapsed_time)
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

                # if classes[7] in fp_set or classes[8] in fp_set or classes[9] in fp_set:
                # 	if th > 0:
                # 		f.write("Th: {0}\nTP: {1}\nFN: {2}\nFP: {3}\nTN: {4}\nGround truth: {5}\nAvg Prediction: {6}\nAll predictions {7}\nVariance {8}\n".
                #             format(th, tp_set, fn_set, fp_set, tn_set, ground_truth, single_prediction, prd, var))

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

        avg_time_for_inference /= (len(test_files) / batch_size)
        # dump all test results to file
        pickle.dump([precision_per_label, recall_per_label, f1_score_per_label, accuracy_per_label, avg_precision,
                     avg_recall, avg_f1, avg_acc, avg_time_for_inference], open(out_pickle, "wb"))
        f.close()

    else:
        # read all test result from file
        precision_per_label, recall_per_label, f1_score_per_label, accuracy_per_label, avg_precision, \
        avg_recall, avg_f1, avg_acc, avg_time_for_inference = pickle.load(open(out_pickle, "rb"))

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
    print "Average inference time for one image: {0}".format(avg_time_for_inference)
    k, v = dict_max_value(avg_acc)
    print "Accuracy with global threshold: {0:.3} - {1:.3}".format(k, v)
    k, v = dict_max_value(avg_f1)
    print "F1 with threshold: {0:.3} - {1:.3}".format(k, v)
    k, v = dict_max_value(avg_recall)
    print "Recall with threshold: {0:.3} - {1:.3}".format(k, v)
    k, v = dict_max_value(avg_precision)
    print "Precision with threshold: {0:.3} - {1:.3}".format(k, v)


class DataGenerator(Iterator):
    def __init__(self, img_folder, annot_folder, filenames, classes,
                 input_shape=None, batch_size=64, shuffle=False, seed=None,
                 data_gen=None, n_per_image=0, category_repr=True, prep="color"):

        self.img_folder = img_folder
        self.annot_folder = annot_folder
        self.filenames = filenames
        self.category_repr = category_repr

        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.classes = classes
        self.data_gen = data_gen
        self.n_per_image = n_per_image
        self.prep = prep
        N = len(self.filenames)

        super(DataGenerator, self).__init__(N, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)

        batch_x = np.zeros((current_batch_size,) + self.input_shape)
        batch_y = np.zeros((current_batch_size, len(self.classes))) if self.category_repr else [0] * current_batch_size
        x = np.zeros((current_batch_size,) + self.input_shape)
        y = np.zeros((current_batch_size, len(self.classes))) if self.category_repr else [0] * current_batch_size

        for i, j in enumerate(index_array):
            if self.prep == 'color':
                im = process_image(
                    os.path.join(self.img_folder, self.filenames[int(j)] + ".jpg"),
                    self.input_shape)
            else:
                im = preprocess_fc(
                    os.path.join(self.img_folder, self.filenames[int(j)] + ".jpg"),
                    self.input_shape)
            batch_x[i] = im
            labels = process_annotation(os.path.join(self.annot_folder, self.filenames[int(j)] + ".xml"))
            batch_y[i] = to_categorical(labels, self.classes) if self.category_repr else set(labels)
            if self.data_gen is not None:
                for num in range(self.n_per_image):
                    x[num] = next(self.data_gen.flow(batch_x[i].reshape((1,) + batch_x[i].shape)))[0].astype(np.uint8)
                    y[num] = batch_y[i]
        np.concatenate((x, batch_x),axis=1)
        np.concatenate((y, batch_y),axis=1)

        return np.array(x, dtype=np.float64), np.array(y)
