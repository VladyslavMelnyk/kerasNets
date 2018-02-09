from __future__ import division

import signal

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Dense

from nets.densenet import DenseNetImageNet169
from datasets.tea_dataset import *

signal.signal(signal.SIGINT, signal.SIG_DFL)

nb_classes = len(classes)

input_shape = (224, 224, 3)

model_name = "densenet_tea_fixed"

net_model = DenseNetImageNet169(input_shape=input_shape, include_top=False, weights='imagenet')
# append classification layer
x = net_model.output
final_output = Dense(nb_classes, activation='sigmoid', name='fc11')(x)

model = Model(inputs=net_model.input, outputs=final_output)
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# load other pre-trined weights
model.load_weights('/data/kerasNets/models/densenet_tea_fixed.hdf5')

# use for training specific layers
for layer in model.layers[:-2]:
	layer.trainable = False

batch_size = 20
nb_epoch = 300

train = read_sets_file(tr_imagesets_folder, "train")
test = read_sets_file(ts_imagesets_folder, "test")

output_dir = "models"

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# callbacks
checkpointer = ModelCheckpoint(filepath=os.path.join(output_dir, model_name + '.hdf5'),
		save_best_only=True, verbose=1)

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=10, min_lr=0.5e-6)

tensorboard = TensorBoard()

#augmentation
data_gen = ImageDataGenerator(
        featurewise_center=False,   # Set input mean to 0 over the dataset, feature-wise.
        samplewise_center=False,    # Set each sample mean to 0.
        featurewise_std_normalization=False,    # Divide inputs by std of the dataset, feature-wise.
        samplewise_std_normalization=False,     # Divide each input by its std.
        zca_whitening=False,    # Apply ZCA whitening epsilon for ZCA whitening.
        zca_epsilon=1e-6,   # Epsilon for ZCA whitening.
        rotation_range=30,  # Degree range for random rotations.
        width_shift_range=0.2,  # Range for random horizontal shifts.
        height_shift_range=0.2,     # Range for random vertical shifts.
        shear_range=0.2,    # Shear angle in counter-clockwise direction as radians.
        zoom_range=0.2,     # Range for random zoom.
        channel_shift_range=0.,     # Range for random channel shifts.
        fill_mode='nearest',    # Points outside the boundaries of the input. Modes:constant/nearest/reflect/wrap
        cval=0.,    # Value used for points outside the boundaries when fill_mode = "constant"
        horizontal_flip=True,   # Randomly flip inputs horizontally.
        vertical_flip=True,     # Randomly flip inputs vertically.
        rescale=None,   # If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided.
        preprocessing_function=None,  # function that will be implied on each input, before any other modification.
        data_format=K.image_data_format()   # One of {"channels_first", "channels_last"}. Default "channels_last".
    )
n_per_image = 10

# train
model.fit_generator(
		DataGenerator(img_folder=tr_image_folder, annot_folder=tr_annotation_folder, filenames=train, classes=classes,
		              shuffle=True, input_shape=input_shape, batch_size=batch_size,
                      data_gen=data_gen, n_per_image=n_per_image, category_repr=True),
		steps_per_epoch=len(train)*n_per_image/batch_size,
		epochs=nb_epoch,
		validation_data=
		DataGenerator(img_folder=ts_image_folder, annot_folder=ts_annotation_folder, filenames=test, classes=classes,
					  input_shape=input_shape, batch_size=batch_size, category_repr=True),
		validation_steps=len(test)/batch_size,
		callbacks=[checkpointer, tensorboard, lr_reducer],
		use_multiprocessing=True,
		verbose=1
)
