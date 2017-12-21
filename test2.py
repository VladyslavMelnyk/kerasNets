from keras.models import load_model

from datasets.dataset import * 

import numpy as np
import os

np.set_printoptions(precision=5, suppress=True)

#switch_to_cpu()

model_location = os.path.join("models", "dense_net_169_original_new_part_retrain.hdf5")
# model_location = os.path.join("models", "dense_net_169_original_retrain.hdf5")
# model2_location = os.path.join("models", "dense_net_169_original.hdf5")

out_dir = "./outdir"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

model = load_model(model_location)
model.summary()

batch_size = 1

input_shape = (500, 300, 3)

th_step = 0.05
thresholds_array = np.arange(0, 1, th_step)

val_aug = ImageDataGenerator()

val_flow = PASCALVOCIterator(directory=root_folder, target_file="test.txt",
                             image_data_generator=val_aug, target_size=(input_shape[0], input_shape[1]),
                             batch_size=batch_size, classes=classes)

test_flow = PASCALVOCIterator(directory=test_folder, target_file="test.txt",
                             image_data_generator=val_aug, target_size=(input_shape[0], input_shape[1]),
                             batch_size=batch_size, classes=classes)


out = model.evaluate_generator(
    val_flow,
    steps=1000
)

print "Validation loss: {0}".format(out)

# val_output = os.path.join(out_dir, "results_val.p")
# val_pred = os.path.join(out_dir, "val.txt")

# model_evaluation(model,
#                  val_flow,
#                  thresholds_array, classes, validation, batch_size, val_output, val_pred)

out = model.evaluate_generator(
    test_flow,
    steps=100
)

print "Test loss: {0}".format(out)

# test_output = os.path.join(out_dir, "results_test.p")
# test_pred = os.path.join(out_dir, "test.txt")
#
# model_evaluation(model,
#                  DataGenerator(img_folder=test_image_folder, annot_folder=test_annotation_folder,
#                                filenames=test, classes=classes, input_shape=input_shape, batch_size=batch_size,
#                                category_repr=False),
#                  thresholds_array, classes, test, batch_size, test_output, test_pred)
