from __future__ import division

import signal

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import load_model

from dataset import *

signal.signal(signal.SIGINT, signal.SIG_DFL)

nb_classes = len(classes)

input_shape = (1000, 600, 3)

model_name = "dense_net_169"

model_location = os.path.join("models", model_name + ".hdf5")

model = load_model(model_location)
model.summary()

optimizer = Adam(lr=1e-5)

model.compile(loss='binary_crossentropy', optimizer=optimizer)

batch_size = 3
nb_epoch = 100

train = read_sets_file(imagesets_folder, "train")
test = read_sets_file(imagesets_folder, "test")

output_dir = "models"

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# callbacks
checkpointer = ModelCheckpoint(
		filepath=os.path.join(output_dir, model_name + '.hdf5'),
		save_best_only=True)


lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=20, min_lr=0.1e-6)

tensorboard = TensorBoard()

# train
model.fit_generator(
		DataGenerator(img_folder=image_folder, annot_folder=annotation_folder, filenames=train, classes=classes,
		              shuffle=True, input_shape=input_shape, batch_size=batch_size, vertical_flip=True,
		              horizontal_flip=True, category_repr=True),
		steps_per_epoch=len(train)*4/batch_size,
		epochs=nb_epoch,
		validation_data=
		DataGenerator(img_folder=image_folder, annot_folder=annotation_folder, filenames=test, classes=classes,
					  input_shape=input_shape, batch_size=batch_size, category_repr=True),
		validation_steps=len(test)/batch_size,
		callbacks=[checkpointer, tensorboard, lr_reducer],
		pickle_safe=True,
		verbose=1
)
