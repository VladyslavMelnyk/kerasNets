from __future__ import division

import os

import cv2
import tensorflow as tf
import numpy as np
from xml.dom import minidom as xml

from keras.preprocessing.image import Iterator
import keras.backend.tensorflow_backend as K


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
		return os.path.dirname(parent_directory(file_path, level-1))


def process_image(image_file, input_shape):
	"""
	Read and reshape image
	:param image_file: path to image
	:param input_shape: new size
	:return: resized image
	"""
	im = cv2.imread(image_file, cv2.IMREAD_COLOR)
	im_resize = cv2.resize(im, dsize=(input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)
	# normalized = (im_resize - img_mean) / img_std
	# return normalized
	return im_resize


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


class DataGenerator(Iterator):

	def __init__(self, img_folder, annot_folder, filenames, classes,
	             input_shape=None, batch_size=64, shuffle=False, seed=None, horizontal_flip=False,
	             vertical_flip=False, category_repr=True):

		self.img_folder = img_folder
		self.annot_folder = annot_folder
		self.filenames = filenames
		self.category_repr = category_repr

		self.input_shape = input_shape
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.classes = classes
		self.horizontal_flip = horizontal_flip
		self.vertical_flip = vertical_flip
		self.extension_koeff = 1 + int(horizontal_flip) + int(vertical_flip) + int(horizontal_flip and vertical_flip)
		N = len(self.filenames) * self.extension_koeff

		super(DataGenerator, self).__init__(N, batch_size, shuffle, seed)

	def next(self):
		with self.lock:
			index_array, _, current_batch_size = next(self.index_generator)

		batch_x = np.zeros((current_batch_size, ) + self.input_shape)
		batch_y = np.zeros((current_batch_size, len(self.classes))) if self.category_repr else [0] * current_batch_size

		for i, j in enumerate(index_array):
			im = process_image(os.path.join(self.img_folder, self.filenames[int(j/self.extension_koeff)] + ".jpg"),
			                   self.input_shape)
			batch_x[i] = self.image_selection(j, im)
			labels = process_annotation(os.path.join(self.annot_folder, self.filenames[int(j/self.extension_koeff)]
			                                        + ".xml"))
			batch_y[i] = to_categorical(labels, self.classes) if self.category_repr else set(labels)

		return batch_x, batch_y

	def image_selection(self, j, im):
		if j % self.extension_koeff == 0:
			# original image
			return im
		elif j % self.extension_koeff == 1:
			# vertical flip
			return cv2.flip(im, 1)
		elif j % self.extension_koeff == 2:
			# horizontal flip
			return cv2.flip(im, 0)
		elif j % self.extension_koeff == 3:
			# horizontal and vertical flip
			return cv2.flip(im, -1)
