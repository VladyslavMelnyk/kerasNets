import os
import cv2
import random
import string
import numpy as np
import yaml
from datasets.utils import *
from nets.densenet import DenseNetImageNet169, DenseNetImageNet121
from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint, TensorBoard, History, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.engine import Model
from keras.layers import Dense, Flatten
from keras import backend as K
from sklearn.model_selection import train_test_split

def preprocess_fc(img, rescale=1. / 255, input_shape=None):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out_img = np.zeros(img.shape)
    out_img[:, :, 0] = img_gray
    out_img[:, :, 1] = img_gray
    out_img[:, :, 2] = img_gray
    if input_shape is not None:
        out_img = cv2.resize(out_img, (input_shape[1], input_shape[0]))
    else:
        out_img = cv2.resize(out_img, (224, 224))
    out_img = out_img * rescale
    return out_img


def preprocess_fconv(img, rescale=1. / 255, input_shape=None):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out_img = np.zeros(img.shape)
    out_img[:, :, 0] = img_gray
    out_img[:, :, 1] = img_gray
    out_img[:, :, 2] = img_gray
    min_size = out_img.shape[0]
    im_scale = float(224) / min_size
    out_img = cv2.resize(out_img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    out_img = out_img * rescale
    return out_img

def preprocess_clr(img, input_shape):
    im_resized = cv2.resize(img, dsize=(input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)
    return im_resized

def display_labels(img, classes, keys):
    font = cv2.FONT_HERSHEY_SIMPLEX
    i = 1
    step = 50
    out_img = img.copy()
    cv2.putText(out_img, 'z - QUIT', (img.shape[1]-150, step), font, 0.7, 255, 2, cv2.LINE_AA)
    for cls_ind in xrange(len(classes)):
        cv2.putText(out_img, str(keys[cls_ind]+'-'+classes[cls_ind]), (0, step * i), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        i += 1
    return out_img

def data_generator(all_images, batch_size=1, classes=None, augmentate=False, n_per_image=0):
    n_classes = len(classes)

    if augmentate:
        imgs = []
        for idx in range(len(all_images)):
            new_img = data_augmentation(idx, all_images, n_per_image)
            [imgs.append(x) for x in new_img]
        [all_images.append(x) for x in imgs]

    while True:
        img_idx = np.random.randint(0, len(all_images), batch_size)
        image_batch = []
        label_batch = []
        for idx in img_idx:
            image_batch.append(all_images[idx][0])
            label = [0] * n_classes
            label[all_images[idx][1]] = 1
            label_batch.append(label)

        yield np.array(image_batch, dtype=np.float64), np.array(label_batch)


def data_augmentation(idx, all_images, n_per_image):
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

    images = []
    for num in range(n_per_image):
        aug_img = next(data_gen.flow(all_images[idx][0].reshape((1,) + all_images[idx][0].shape)))[0].astype(np.uint8)
        images.append((aug_img, all_images[idx][1]))
    return images

def load_images(dataset_dir, classes, network_type, input_shape=None):
    all_images = []
    if network_type == 'fconv':
        preprocess_fn = preprocess_fconv
    elif network_type == 'fc':
        preprocess_fn = preprocess_fc
    elif network_type == 'fc_clr':
        preprocess_fn = preprocess_clr
    else:
        preprocess_fn = None
    classes_idx = dict(zip(classes, range(len(classes))))
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        images = []
        for f in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, f)
            img = cv2.imread(img_path)
            img_data = preprocess_fn(img, input_shape=input_shape)
            images.append((img_data, classes_idx[cls]))
        all_images.extend(images)
        print("Class:{}, images:{}".format(cls, len(images)))
    print("All images:{}".format(len(all_images)))
    return all_images

def save_image(im, path, cls):
	if path is None:
		path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

	if not os.path.exists(path):
		os.mkdir(path)

	timestamp = str(datetime.datetime.now()).split(" ")
	filename = "img_" + timestamp[0] + "_" + timestamp[1] + ".jpg"
	cv2.imwrite(os.path.join(path, cls, filename), im)

def create_image_lists(all_images, val_pts=0.2):
    """Splites a list of images into train and validation sets.

    :param all_images: List of all images provided for trainig.
    :param validation_pct: Integer percentage of images reserved for validation.
    """
    train_images = []
    val_images = []
    X = [i[0] for i in all_images]
    y = [i[1] for i in all_images]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=val_pts)
    for idx in range(len(X_train)):
        train_images.append((X_train[idx],y_train[idx]))
    for idx in range(len(X_test)):
        val_images.append((X_test[idx],y_test[idx]))
    print "Train set size = {} images".format(len(train_images))
    print "Validation set size = {} images".format(len(val_images))
    print "Finished processing all images"
    return train_images, val_images

def build_model(arch, no_classes, network_type, optimizer='adam', pooling='avg', weights='imagenet', input_shape=None):
   if arch == 'resnet50':
       if network_type == 'fc' or network_type == "fc_clr":
           model = ResNet50(input_shape=input_shape, include_top=False, weights=weights)
           x = Flatten()(model.output)
       else:
           model = ResNet50(input_shape=(None, None, 3), include_top=False, pooling=pooling, weights=weights)
           x = model.output
   elif arch == 'densenet':
       if network_type == 'fc' or network_type == "fc_clr":
           model = DenseNetImageNet169(input_shape=input_shape, include_top=False, weights=weights)
           x = model.output
       else:
           model = DenseNetImageNet169(input_shape=(None, None, 3), include_top=False, pooling=pooling, weights=weights)
           x = model.output
   elif arch == 'densenet121':
       if network_type == 'fc' or network_type == "fc_clr":
           model = DenseNetImageNet121(input_shape=input_shape, include_top=False, weights=weights)
           x = model.output
       else:
           model = DenseNetImageNet121(input_shape=(None, None, 3), include_top=False, pooling=pooling, weights=weights)
           x = model.output
   x = Dense(no_classes, activation='softmax')(x)
   model = Model(model.inputs, x)
   model.compile(optimizer=optimizer, loss='categorical_crossentropy')
   return model

class ClassifierTool(yaml.YAMLObject):
    yaml_tag = u'Classifier'

    def __init__(self):
        super(ClassifierTool, self).__init__()

    def train(self, outdir, epochs=100, batch_size=7, finetune=False, init_weights=None, augmentate=False):
        """

        :param outdir: Output director to save models and training info.
        :param epochs: Total epochs.(default 100)
        :param batch_size: Batch size
        :param init_weights: Weights path for fine tuning.
        """

        print "Starting training with such parameters:"
        print "Model name={}".format(self.name)
        print "Model location={}".format(self.model_location)
        print "Dataset location={}".format(self.dataset_dir)
        print "Classes={}".format(self.classes)
        print "Input shape={}".format(self.input_shape)
        print "Network arch={}".format(self.arch)
        print "NN type={}".format(self.network_type)
        print "Output dir={}".format(outdir)
        print "Batch size={}".format(batch_size)
        print "Number epochs={}".format(epochs)
        print "Initial weighs={}".format(init_weights)
        print "Using image augmentation={}".format(augmentate)

        all_images = load_images(self.dataset_dir, self.classes, self.network_type, input_shape=self.input_shape)
        train_images, val_images = create_image_lists(all_images)
        n_per_img = 50
        train_generator = data_generator(train_images, batch_size, classes=self.classes, augmentate=augmentate, n_per_image=n_per_img)
        val_generator = data_generator(val_images, batch_size=1, classes=self.classes)
        #switch_to_cpu()
        set_fraction_of_gpu_memory(0.85)
        model = build_model(arch=self.arch, no_classes=len(self.classes), network_type=self.network_type, input_shape=self.input_shape)
        model.summary(line_length=120)
        if init_weights is not None:
            model.load_weights(str(init_weights))
            print 'Pre-trained weighs loaded successfully'

        if finetune:
            for layer in model.layers[:-1]:
                layer.trainable = False

        callbacks = [ModelCheckpoint(os.path.join(outdir, '{}_model.hdf5'.format(self.name)), save_best_only=True, verbose=1),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=10, min_lr=0.5e-6),
                        TensorBoard(os.path.join(outdir, 'tf_logs')), History()]

        model.fit_generator(train_generator, steps_per_epoch=int(len(train_images*n_per_img)/batch_size), epochs=epochs,
                            validation_data=val_generator, validation_steps=len(val_images),
                            callbacks=callbacks, use_multiprocessing=True, verbose=1)
        # class_weight={0: 1.0, 1: 1.3})

    def test(self, dir=None, model_path=None):
        """
        Evaluate model on dataset.
        :param dir: Evaluation dataset dir.
        :param model_path: Path of hdf file.
        """
        print "Starting evaluation with such parameters:"
        print "Model name={}".format(self.name)
        print "Classes={}".format(self.classes)
        print "Input shape={}".format(self.input_shape)
        print "Network arch={}".format(self.arch)
        print "NN type={}".format(self.network_type)

        if dir is not None:
            print "Testing on dataset={}".format(dir)
            all_images = load_images(dir, self.classes, self.network_type, input_shape=self.input_shape)
        else:
            print "Testing on default dataset"
            all_images = load_images(self.dataset_dir, self.classes, self.network_type, input_shape=self.input_shape)

        #switch_to_cpu()
        #set_fraction_of_gpu_memory(0.85)
        model = build_model(arch=self.arch, no_classes=len(self.classes), network_type=self.network_type, input_shape=self.input_shape)

        if model_path is not None:
            model.load_weights(model_path)
            print "Using model={}".format(model_path)
        else:
            model.load_weights(self.model_location)
            print "Using model={}".format(self.model_location)
        print "Test starting, please wait..."
        nb_classes = len(self.classes)
        fp = dict(zip(self.classes, len(self.classes) * [0]))
        ac = dict(zip(self.classes, len(self.classes) * [0]))
        det = dict(zip(self.classes, len(self.classes) * [0]))
        for img in all_images:
            class_prob = np.squeeze(model.predict(np.expand_dims(img[0], 0)))
            cls_ind = np.argmax(class_prob)
            class_prob = np.max(class_prob)
            class_gt = self.classes[img[1]]
            class_pred = self.classes[cls_ind]
            if class_pred != class_gt:
                fp[class_gt] += 1
                print('Prob:{}, gt:{}, pred:{}'.format(class_prob, class_gt, class_pred))
            det[class_gt] += 1
        for i in xrange(nb_classes):
            class_gt = self.classes[i]
            ac[class_gt] = (det[class_gt] - fp[class_gt])/float(det[class_gt])
            ac[class_gt] = float("{0:.2f}".format(ac[class_gt]))
        acc = np.mean(ac.values())
        print("False detections :{}".format(fp))
        print("Class accuracy :{}".format(ac))
        print("Average accuracy : {}".format(acc))

    def annotate(self, source, outdir=None):
        if source == "web_cam":
            from lv.frame_grabbers.webcam_grabber import WebCamFrameGrabber as FrameReader
        elif source == "ids":
            from lv.frame_grabbers.ids_frame_grabber import IDSFrameReader as FrameReader
        elif source == "pyro":
            from lv.frame_grabbers.frame_grabber_client import FrameReader
        else:
            raise NotImplementedError("Unsupported source: {0}".format(source))

        if outdir is None:
            outdir = self.dataset_dir
            print "Images will be added to default dataset"

        cam = FrameReader()
        cam.start()
        keys = list(string.ascii_lowercase)

        while True:
            frame = cam.get_frame()

            img = frame.copy()
            frame = display_labels(frame, self.classes, keys)
            cv2.namedWindow('Annotate', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Annotate', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Annotate', frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            for idx in xrange(len(self.classes)):
                if key == ord(str(keys[idx])):
                    save_image(img, outdir, self.classes[idx])

        cam.stop()
        cv2.destroyAllWindows