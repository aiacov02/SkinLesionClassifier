import tensorflow as tf



import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.python.keras import applications
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.densenet import DenseNet121

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


import math
import cv2
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


config = tf.ConfigProto()


config.gpu_options.allow_growth = True
session = tf.Session(config=config)




from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# dimensions of our images.
img_width, img_height = 224, 224

# number of epochs to train top model
epochs = 100
# batch size used by flow_from_directory and predict_generator
batch_size = 32

df = pd.read_csv(r"./train.csv")

df['image_id'] = df['image_id'].astype(str) + '.jpg'


start_dir = "start_dir/"

# The paths for the training and validation images
train_path = start_dir + 'base_dir/train_dir'
valid_path = start_dir + 'base_dir/val_dir'
test_path = start_dir + 'base_dir/test_dir'

# Declare a few useful values
num_train_samples = 64412
num_val_samples = 9548
num_test_samples = 802
train_batch_size = 32
val_batch_size = 32
test_batch_size = 32
image_size = 224

# Declare how many steps are needed in an iteration
train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)
test_steps = np.ceil(num_test_samples / test_batch_size)


####################################################
# Data Visualization


def visualize_data():
    count_df = df['dx'].value_counts()
    print(count_df)
    try:
        # count_df = df['dx'].value_counts()
        ax = count_df.plot.bar(x='Skin Lesion Type', y='Count', rot=0)
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    except:
        print("Display Error")



def train_top_model_vgg():


    # VGG16 generators

    # Set up generators
    train_batches = ImageDataGenerator(
        preprocessing_function= \
            applications.vgg16.preprocess_input).flow_from_directory(
        train_path,
        target_size=(image_size, image_size),
        batch_size=train_batch_size)

    valid_batches = ImageDataGenerator(
        preprocessing_function= \
            applications.vgg16.preprocess_input).flow_from_directory(
        valid_path,
        target_size=(image_size, image_size),
        batch_size=val_batch_size)

    test_batches = ImageDataGenerator(
        preprocessing_function= \
            applications.vgg16.preprocess_input).flow_from_directory(
        test_path,
        target_size=(image_size, image_size),
        batch_size=test_batch_size,
        shuffle=False)

    my_model = applications.VGG16(include_top=True, weights='imagenet')


    initial_model = Sequential()
    for layer in my_model.layers[:22]:
        initial_model.add(layer)

    initial_model.layers.pop()

    initial_model.add(Dense(1024, activation='relu'))
    initial_model.add(Dropout(0.5))
    initial_model.add(Dense(7, activation="softmax"))


    print(initial_model.summary())


    # Define Top2 and Top3 Accuracy

    def top_3_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=3)

    def top_2_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=2)


    initial_model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=True),
                          loss="categorical_crossentropy",
                          metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy, "accuracy"])


    filepath = "modelVGG.h5"

    # Declare a checkpoint to save the best version of the model
    checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1,
                                 save_best_only=True, mode='max')

    # Reduce the learning rate as the learning stagnates
    reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=2,
                                  verbose=1, mode='max', min_lr=0.00001)

    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=5,
                                   verbose=1, mode='max')

    callbacks_list = [checkpoint, reduce_lr, early_stopping]

    history = initial_model.fit_generator(
        train_batches,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=valid_batches,
        validation_steps=val_steps,
        callbacks=callbacks_list)


    try:

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("VGG16_accuracy_training_plot.png")
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("VGG16_loss_training_plot.png")

    except:
        print("Error")


def train_top_model_resnet():

    # ResNet50 generators

    # Set up generators
    train_batches = ImageDataGenerator(
        preprocessing_function= \
            applications.resnet50.preprocess_input).flow_from_directory(
        train_path,
        target_size=(image_size, image_size),
        batch_size=train_batch_size)

    valid_batches = ImageDataGenerator(
        preprocessing_function= \
            applications.resnet50.preprocess_input).flow_from_directory(
        valid_path,
        target_size=(image_size, image_size),
        batch_size=val_batch_size)

    test_batches = ImageDataGenerator(
        preprocessing_function= \
            applications.resnet50.preprocess_input).flow_from_directory(
        test_path,
        target_size=(image_size, image_size),
        batch_size=test_batch_size,
        shuffle=False)

    input_tensor = Input(shape = (224,224,3))

    #Loading the model
    model = ResNet50(input_tensor= input_tensor,weights='imagenet',include_top=False)


    x = model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(7,activation ='softmax')(x)

    model = Model(inputs=model.input, outputs=predictions)


    def top_3_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=3)

    def top_2_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=2)

    model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False),
                          loss="categorical_crossentropy",
                          metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy, "accuracy"])

    print(model.summary())

    # Declare a checkpoint to save the best version of the model
    checkpoint = ModelCheckpoint("modelResNet2.h5", monitor='val_categorical_accuracy', verbose=1,
                                 save_best_only=True, mode='max')

    # Reduce the learning rate as the learning stagnates
    reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=2,
                                  verbose=1, mode='max', min_lr=0.00001)

    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=10, verbose=1, mode='max')

    callbacks_list = [checkpoint, reduce_lr,
                      early_stopping
                      ]

    history = model.fit_generator(train_batches, epochs=epochs, shuffle=True, validation_data = valid_batches, steps_per_epoch=train_steps, validation_steps = val_steps,  verbose=1, callbacks=callbacks_list)


    try:

        # # Evaluation of the best epoch
        model.load_weights('modelResNet.h5')

        val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = \
            model.evaluate_generator(valid_batches, steps=val_steps)

        print('val_loss:', val_loss)
        print('val_cat_acc:', val_cat_acc)
        print('val_top_2_acc:', val_top_2_acc)
        print('val_top_3_acc:', val_top_3_acc)

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("ResNet50_accuracy_training_plot.png")
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("ResNet50_loss_training_plot.png")
    except :
        print("Error")


def train_top_model_inceptionV3():


    # InceptionV3 generators

    # Set up generators
    train_batches = ImageDataGenerator(
        preprocessing_function= \
            applications.inception_v3.preprocess_input).flow_from_directory(
        train_path,
        target_size=(299, 299),
        batch_size=train_batch_size)

    valid_batches = ImageDataGenerator(
        preprocessing_function= \
            applications.inception_v3.preprocess_input).flow_from_directory(
        valid_path,
        target_size=(299, 299),
        batch_size=val_batch_size)

    test_batches = ImageDataGenerator(
        preprocessing_function= \
            applications.inception_v3.preprocess_input).flow_from_directory(
        test_path,
        target_size=(299, 299),
        batch_size=test_batch_size,
        shuffle=False)


    input_tensor = Input(shape = (299,299,3))

    #Loading the model
    model = InceptionV3(input_tensor= input_tensor,weights='imagenet',include_top=False)


    # add a global spatial average pooling layer
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(7, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=model.input, outputs=predictions)


    def top_3_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=3)

    def top_2_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=2)


    model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False),
                          loss="categorical_crossentropy",
                          metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy, "accuracy"])

    print(model.summary())

    # Declare a checkpoint to save the best version of the model
    checkpoint = ModelCheckpoint("modelInceptionV3.h5", monitor='val_categorical_accuracy', verbose=1,
                                 save_best_only=True, mode='max')

    # Reduce the learning rate as the learning stagnates
    reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=2,
                                  verbose=1, mode='max', min_lr=0.00001)

    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=10, verbose=1, mode='max')

    callbacks_list = [checkpoint, reduce_lr,
                      early_stopping
                      ]

    history = model.fit_generator(train_batches, epochs=epochs, shuffle=True, validation_data = valid_batches, steps_per_epoch=train_steps, validation_steps = val_steps,  verbose=1, callbacks=callbacks_list)


    try:
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("InceptionV3_accuracy_training_plot.png")
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("InceptionV3_loss_training_plot.png")
    except:
        print("Error")


def train_top_model_densenet121():



    # DesneNet generators

    # Set up generators
    train_batches = ImageDataGenerator(
        preprocessing_function= \
            applications.densenet.preprocess_input).flow_from_directory(
        train_path,
        target_size=(image_size, image_size),
        batch_size=train_batch_size)

    valid_batches = ImageDataGenerator(
        preprocessing_function= \
            applications.densenet.preprocess_input).flow_from_directory(
        valid_path,
        target_size=(image_size, image_size),
        batch_size=val_batch_size)

    test_batches = ImageDataGenerator(
        preprocessing_function= \
            applications.densenet.preprocess_input).flow_from_directory(
        test_path,
        target_size=(image_size, image_size),
        batch_size=test_batch_size,
        shuffle=False)

    input_tensor = Input(shape = (224,224,3))

    #Loading the model
    model = DenseNet121(input_tensor= input_tensor,weights='imagenet',include_top=False)

    # add a global spatial average pooling layer
    x = model.output
    x = GlobalAveragePooling2D()(x)
    # add relu layer
    x = Dense(1024, activation='relu')(x)
    # and a softmax layer for 7 classes
    predictions = Dense(7, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=model.input, outputs=predictions)


    def top_3_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=3)

    def top_2_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=2)


    model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False),
                          loss="categorical_crossentropy",
                          metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])

    print(model.summary())

    # Declare a checkpoint to save the best version of the model
    checkpoint = ModelCheckpoint("modelDenseNet121.h5", monitor='val_categorical_accuracy', verbose=1,
                                 save_best_only=True, mode='max')

    # Reduce the learning rate as the learning stagnates
    reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=2,
                                  verbose=1, mode='max', min_lr=0.00001)

    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=10, verbose=1, mode='max')

    callbacks_list = [checkpoint, reduce_lr,
                      early_stopping
                      ]

    history = model.fit_generator(train_batches,
                                  # class_weight = class_weights,
                                  epochs=epochs, shuffle=True, validation_data = valid_batches, steps_per_epoch=train_steps, validation_steps = val_steps,  verbose=1, callbacks=callbacks_list)


    # # Evaluation of the best epoch
    model.load_weights('modelDenseNet.h5')

    val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = \
        model.evaluate_generator(valid_batches, steps=val_steps)

    print('val_loss:', val_loss)
    print('val_cat_acc:', val_cat_acc)
    print('val_top_2_acc:', val_top_2_acc)
    print('val_top_3_acc:', val_top_3_acc)


# visualize_data()
train_top_model_resnet()
train_top_model_vgg()
train_top_model_inceptionV3()
train_top_model_densenet121()
