from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, Flatten

from datasets.dataset import * 

import numpy as np
import os

np.set_printoptions(precision=5, suppress=True)

#switch_to_cpu()

model_location = os.path.join("models", "res_net50.hdf5")
# model_location = os.path.join("models", "dense_net_169_original_retrain.hdf5")
# model2_location = os.path.join("models", "dense_net_169_original.hdf5")

out_dir = "./outdir"

nb_classes = len(classes)

model = load_model(model_location)

batch_size = 1

input_shape = (600, 400, 3)

th_step = 0.05
thresholds_array = np.arange(0, 1, th_step)

# val_aug = ImageDataGenerator()
#
# val_flow = PASCALVOCIterator(directory=root_folder, target_file="test.txt",
#                              image_data_generator=val_aug, target_size=(input_shape[0], input_shape[1]),
#                              batch_size=batch_size, classes=classes)
#
# test_flow = PASCALVOCIterator(directory=test_folder, target_file="test.txt",
#                              image_data_generator=val_aug, target_size=(input_shape[0], input_shape[1]),
#                              batch_size=batch_size, classes=classes)


# out = model.evaluate_generator(
#     val_flow,
#     steps=1000
# )
#
# print "Validation loss: {0}".format(out)

val_output = os.path.join(out_dir, "results_val.p")
if os.path.exists(val_output):
    os.remove(val_output)
val_pred = os.path.join(out_dir, "val.txt")
if os.path.exists(val_pred):
    os.remove(val_pred)

model_evaluation(model,
                 DataGenerator(img_folder=train_val_image_folder, annot_folder=train_val_annotation_folder,
                               filenames=validation, classes=classes, input_shape=input_shape, batch_size=batch_size,
                               category_repr=False),
                 thresholds_array, classes, validation, batch_size, val_output, val_pred)

# out = model.evaluate_generator(
#     test_flow,
#     steps=100
# )
#
# print "Test loss: {0}".format(out)

test_output = os.path.join(out_dir, "results_test.p")
if os.path.exists(test_output):
    os.remove(test_output)
test_pred = os.path.join(out_dir, "test.txt")
if os.path.exists(test_pred):
    os.remove(test_pred)

model_evaluation(model,
                 DataGenerator(img_folder=test_image_folder, annot_folder=test_annotation_folder,
                               filenames=test, classes=classes, input_shape=input_shape, batch_size=batch_size,
                               category_repr=False),
                 thresholds_array, classes, test, batch_size, test_output, test_pred)
