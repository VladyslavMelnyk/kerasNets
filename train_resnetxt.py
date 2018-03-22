from __future__ import division

import signal

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD, Adadelta, Adam
from nets.resnext import ResNext,ResNextImageNet

from utils import set_fraction_of_gpu_memory
from keras.models import load_model
from dataset import *

import cv2

signal.signal(signal.SIGINT, signal.SIG_DFL)

nb_classes = len(classes)

input_shape = (500, 300, 3)

model_name = "resnetxt34"
model_location = os.path.join("models", model_name + ".hdf5")



#train

net_model = ResNextImageNet (input_shape=input_shape, include_top=False, weights=None, pooling='avg')
# append classification layer
x = net_model.output
final_output = Dense(nb_classes, activation='sigmoid', name='fc11')(x)
model = Model(inputs=net_model.input, outputs=final_output)

#finetune
model.load_weights(model_location)

model.summary()

#Prediction
img = cv2.imread("/data/rfcn-object-detection/data/skoda/train/JPEGImages/20170718_125544_Richtone(HDR).jpg")
yPreds = model.predict(img)
yPred = np.argmax(yPreds, axis=1)
print yPred

"""
optimizer = SGD(lr=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

batch_size = 4
nb_epoch = 100
nb_epoch1 = 50

train = read_sets_file(imagesets_folder, "train")
test = read_sets_file(imagesets_folder, "test")

output_dir = "models"

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# callbacks
checkpointer = ModelCheckpoint(
		filepath=os.path.join(output_dir, model_name + '.hdf5'),
		save_best_only=True)


lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=20, min_lr=0.5e-6)

tensorboard = TensorBoard()

# train
set_fraction_of_gpu_memory(0.85)
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
)"""

