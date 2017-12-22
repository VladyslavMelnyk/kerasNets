from __future__ import division

import signal

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, Flatten

from datasets.skoda_dataset import *

# TODO: Add dataset normalization

signal.signal(signal.SIGINT, signal.SIG_DFL)

model_name = 'res_net50'

nb_classes = len(classes)

input_shape = (600, 400, 3)

net_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

# append classification layer
x = net_model.output
x = Flatten()(x)
final_output = Dense(nb_classes, activation='sigmoid', name='fc11')(x)

model = Model(inputs=net_model.input, outputs=final_output)
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam')

batch_size = 4
nb_epoch = 100

train = read_sets_file(imagesets_folder, "train")
test = read_sets_file(imagesets_folder, "test")

output_dir = "models"

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# callbacks
checkpointer = ModelCheckpoint(
		filepath=os.path.join(output_dir, model_name + '.hdf5'),
		monitor='val_loss', verbose=1, save_best_only=True)

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=20, min_lr=0.5e-6)

# early_stop = EarlyStopping(patience=5)
tensorboard = TensorBoard()

# train
model.fit_generator(
		DataGenerator(img_folder=image_folder, annot_folder=annotation_folder, filenames=train, classes=classes, shuffle=True,
		              input_shape=input_shape, batch_size=batch_size, vertical_flip=True, horizontal_flip=True),
		steps_per_epoch=len(train)*4/batch_size,
		epochs=nb_epoch,
		validation_data=
		DataGenerator(img_folder=image_folder, annot_folder=annotation_folder, filenames=test, classes=classes,
		              input_shape=input_shape, batch_size=batch_size),
		validation_steps=len(test)/batch_size,
		callbacks=[checkpointer, tensorboard, lr_reducer],
		pickle_safe=True,
		verbose=1
)
