from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, Flatten

from datasets.tea_dataset import *

import numpy as np
import os

np.set_printoptions(precision=5, suppress=True)

#switch_to_cpu()

model_location = os.path.join("models", "densenet_tea_color_new.hdf5")
# model_location = os.path.join("models", "dense_net_169_original_retrain.hdf5")
# model2_location = os.path.join("models", "dense_net_169_original.hdf5")

out_dir = "./outdir"

nb_classes = len(classes)

model = load_model(model_location)

batch_size = 1

input_shape = (300, 500, 3)

th_step = 0.05
thresholds_array = np.arange(0, 1, th_step)

val_output = os.path.join(out_dir, "results_val.p")
if os.path.exists(val_output):
    os.remove(val_output)
val_pred = os.path.join(out_dir, "val.txt")
if os.path.exists(val_pred):
    os.remove(val_pred)

model_evaluation(model,
                 DataGenerator(img_folder=ts_image_folder, annot_folder=ts_annotation_folder,
                               filenames=validation, classes=classes, input_shape=input_shape, batch_size=batch_size,
                               category_repr=False, mode="color", box_folder=box_annotation_folder_ts),
                 thresholds_array, classes, validation, batch_size, val_output, val_pred)

test_output = os.path.join(out_dir, "results_train.p")
if os.path.exists(test_output):
    os.remove(test_output)
test_pred = os.path.join(out_dir, "train.txt")
if os.path.exists(test_pred):
    os.remove(test_pred)

model_evaluation(model,
                 DataGenerator(img_folder=tr_image_folder, annot_folder=tr_annotation_folder,
                               filenames=train, classes=classes, input_shape=input_shape, batch_size=batch_size,
                               category_repr=False, mode="color", box_folder=box_annotation_folder),
                 thresholds_array, classes, train, batch_size, test_output, test_pred)
