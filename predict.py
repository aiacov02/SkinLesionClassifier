import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.python.keras import applications
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, recall_score, roc_auc_score, precision_score, f1_score
import itertools
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.densenet import DenseNet121

from tensorflow.python.keras.utils.vis_utils import plot_model




from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


import math
import cv2
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

config = tf.ConfigProto()


config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

# number of epochs to train top model
epochs = 100
# batch size used by flow_from_directory and predict_generator
batch_size = 16

df = pd.read_csv(r"./train.csv")

df['image_id'] = df['image_id'].astype(str) + '.jpg'

start_dir = "start_dir/"


test2_path = start_dir + 'base_dir/test2_dir'

# Declare a few useful values

num_test_samples = 161
train_batch_size = 16
val_batch_size = 16
test_batch_size = 16
image_size = 224

# Declare how many steps are needed in an iteration
train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)
test_steps = np.ceil(num_test_samples / test_batch_size)


def predict_vgg():

    # VGG16 generators

    # Set up generators
    test_batches = ImageDataGenerator(
        preprocessing_function= \
            applications.vgg16.preprocess_input).flow_from_directory(
        test2_path,
        target_size=(image_size, image_size),
        batch_size=test_batch_size,
        shuffle=False)


    my_model = applications.VGG16(include_top=True, weights='imagenet')


    initial_model = Sequential()
    for layer in my_model.layers[:22]:
        initial_model.add(layer)

    initial_model.layers.pop()


    for layer in initial_model.layers:
        layer.trainable = False

    initial_model.add(Dense(1024, activation='relu'))
    initial_model.add(Dropout(0.5))
    initial_model.add(Dense(7, activation="softmax"))


    initial_model.load_weights('modelVGG.h5')

    model = initial_model

    print(model.summary())

    plot_model(model, to_file="VGG16Model.png", show_shapes=True)

    test_batches.reset()

    test_labels = test_batches.classes


    # Make predictions
    predictions = model.predict_generator(test_batches, steps=test_steps, verbose=1)

    # Declare a function for plotting the confusion matrix
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_VGG16.png')

    cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

    cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']

    plot_confusion_matrix(cm, cm_plot_labels)

    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)

    print("Mean recall " + str(np.mean(recall).item()))
    print("Mean precision " + str(np.mean(precision).item()))

    print("Mean recall " + str(recall_score(test_labels, predictions.argmax(axis=1), average='weighted')))
    print("Balanced Accuracy " + str(balanced_accuracy_score(test_labels, predictions.argmax(axis=1))))
    print("Mean Precision " + str(precision_score(test_labels, predictions.argmax(axis=1), average='weighted')))
    print("Mean f1 score " + str(f1_score(test_labels, predictions.argmax(axis=1), average='weighted')))


    file = open("VGG16_results.txt","w+")
    file.write("Mean recall " + str(np.mean(recall).item()) + "\n")
    file.write("Mean precision " + str(np.mean(precision).item())+ "\n")
    file.write("Mean recall " + str(recall_score(test_labels, predictions.argmax(axis=1), average='weighted')) + "\n")
    file.write("Balanced Accuracy " + str(balanced_accuracy_score(test_labels, predictions.argmax(axis=1)))+ "\n")
    file.write("Mean Precision " + str(precision_score(test_labels, predictions.argmax(axis=1), average='weighted')) + "\n")
    file.write("Mean f1 score " + str(f1_score(test_labels, predictions.argmax(axis=1), average='weighted')) + "\n")

    file.close()

    # print("Roc AUC score: " + str(roc_auc_score(test_labels, predictions.argmax(axis=1))))

    predicted_class_indices = np.argmax(predictions,axis=1)

    labels = train_batches.class_indices
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    akiec_predictions = labels[0]

    filenames=test_batches.filenames
    results=pd.DataFrame({"Filename":filenames,
                          "Vgg_Predictions":predictions})
    results.to_csv("test_prediction_vgg16.csv", index=False)




def predict_resnet():

    input_tensor = Input(shape = (224,224,3))

    #Loading the model
    model = ResNet50(input_tensor= input_tensor,weights='imagenet',include_top=False)

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(7, activation ='softmax')(x)

    model = tf.keras.models.Model(model.input, x)

    model.load_weights('modelResNet.h5')


    test_batches.reset()

    test_labels = test_batches.classes

    # Make predictions
    predictions = model.predict_generator(test_batches, steps=test_steps, verbose=1)

    # Declare a function for plotting the confusion matrix
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_ResNet50.png')

    cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

    cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']

    plot_confusion_matrix(cm, cm_plot_labels)

    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)

    print("Mean recall " + str(np.mean(recall).item()))
    print("Mean precision " + str(np.mean(precision).item()))

    print("Mean recall " + str(recall_score(test_labels, predictions.argmax(axis=1), average='weighted')))
    print("Balanced Accuracy " + str(balanced_accuracy_score(test_labels, predictions.argmax(axis=1))))
    print("Mean Precision " + str(precision_score(test_labels, predictions.argmax(axis=1), average='weighted')))
    print("Mean f1 score " + str(f1_score(test_labels, predictions.argmax(axis=1), average='weighted')))


    file = open("ResNet50_results.txt","w+")
    file.write("Mean recall " + str(np.mean(recall).item()) + "\n")
    file.write("Mean precision " + str(np.mean(precision).item())+ "\n")
    file.write("Mean recall " + str(recall_score(test_labels, predictions.argmax(axis=1), average='weighted')) + "\n")
    file.write("Balanced Accuracy " + str(balanced_accuracy_score(test_labels, predictions.argmax(axis=1)))+ "\n")
    file.write("Mean Precision " + str(precision_score(test_labels, predictions.argmax(axis=1), average='weighted')) + "\n")
    file.write("Mean f1 score " + str(f1_score(test_labels, predictions.argmax(axis=1), average='weighted')) + "\n")

    file.close()

    # print("Roc AUC score: " + str(roc_auc_score(test_labels, predictions.argmax(axis=1))))

    predicted_class_indices=np.argmax(predictions,axis=1)

    labels = train_batches.class_indices
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    filenames=test_batches.filenames
    results=pd.DataFrame({"Filename":filenames,
                          "ResNet50_Predictions":predictions})
    results.to_csv("test_prediction_resnet50.csv", index=False)

def predict_inceptionv3():

    # InceptionV3 generators

    test_batches = ImageDataGenerator(
        preprocessing_function= \
            applications.inception_v3.preprocess_input).flow_from_directory(
        test2_path,
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

    model.load_weights('modelInceptionV3.h5')



    test_batches.reset()

    test_labels = test_batches.classes

    # Make predictions

    predictions = model.predict_generator(test_batches, steps=test_steps, verbose=1)

    # Declare a function for plotting the confusion matrix
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_InceptionV3.png')

    cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

    cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']

    plot_confusion_matrix(cm, cm_plot_labels)

    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)

    print("Mean recall " + str(np.mean(recall).item()))
    print("Mean precision " + str(np.mean(precision).item()))

    print("Mean recall " + str(recall_score(test_labels, predictions.argmax(axis=1), average='weighted')))
    print("Balanced Accuracy " + str(balanced_accuracy_score(test_labels, predictions.argmax(axis=1))))
    print("Mean Precision " + str(precision_score(test_labels, predictions.argmax(axis=1), average='weighted')))
    print("Mean f1 score " + str(f1_score(test_labels, predictions.argmax(axis=1), average='weighted')))


    file = open("InceptionV3_results.txt","w+")
    file.write("Mean recall " + str(np.mean(recall).item()) + "\n")
    file.write("Mean precision " + str(np.mean(precision).item())+ "\n")
    file.write("Mean recall " + str(recall_score(test_labels, predictions.argmax(axis=1), average='weighted')) + "\n")
    file.write("Balanced Accuracy " + str(balanced_accuracy_score(test_labels, predictions.argmax(axis=1)))+ "\n")
    file.write("Mean Precision " + str(precision_score(test_labels, predictions.argmax(axis=1), average='weighted')) + "\n")
    file.write("Mean f1 score " + str(f1_score(test_labels, predictions.argmax(axis=1), average='weighted')) + "\n")

    file.close()

    predicted_class_indices=np.argmax(predictions,axis=1)

    labels = train_batches.class_indices
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    filenames = test_batches.filenames
    results=pd.DataFrame({"Filename":filenames,
                          "InceptionV3_Predictions":predictions})
    results.to_csv("test_prediction_inceptionv3.csv", index=False)


def predict_densenet121():

    # DesneNet generators

    test_batches = ImageDataGenerator(
        preprocessing_function= \
            applications.densenet.preprocess_input).flow_from_directory(
        test2_path,
        target_size=(image_size, image_size),
        batch_size=test_batch_size,
        shuffle=False)


    input_tensor = Input(shape = (224,224,3))

    #Loading the model
    model = DenseNet121(input_tensor= input_tensor,weights='imagenet',include_top=False)


    # add a global spatial average pooling layer
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(7, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=model.input, outputs=predictions)

    model.load_weights('modelDenseNet121.h5')


    test_batches.reset()

    test_labels = test_batches.classes


    # Make predictions
    predictions = model.predict_generator(test_batches, steps=test_steps, verbose=1)


    # Declare a function for plotting the confusion matrix
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_DenseNet121.png')

    cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

    cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']

    plot_confusion_matrix(cm, cm_plot_labels)

    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)

    print("Mean recall " + str(np.mean(recall).item()))
    print("Mean precision " + str(np.mean(precision).item()))

    print("Mean recall " + str(recall_score(test_labels, predictions.argmax(axis=1), average='weighted')))
    print("Balanced Accuracy " + str(balanced_accuracy_score(test_labels, predictions.argmax(axis=1))))
    print("Mean Precision " + str(precision_score(test_labels, predictions.argmax(axis=1), average='weighted')))
    print("Mean f1 score " + str(f1_score(test_labels, predictions.argmax(axis=1), average='weighted')))


    file = open("DenseNet121_results_last","w+")
    file.write("Mean recall " + str(np.mean(recall).item()) + "\n")
    file.write("Mean precision " + str(np.mean(precision).item())+ "\n")
    file.write("Mean recall " + str(recall_score(test_labels, predictions.argmax(axis=1), average='weighted')) + "\n")
    file.write("Balanced Accuracy " + str(balanced_accuracy_score(test_labels, predictions.argmax(axis=1)))+ "\n")
    file.write("Mean Precision " + str(precision_score(test_labels, predictions.argmax(axis=1), average='weighted')) + "\n")
    file.write("Mean f1 score " + str(f1_score(test_labels, predictions.argmax(axis=1), average='weighted')) + "\n")

    file.close()

    predicted_class_indices=np.argmax(predictions,axis=1)


    labels = train_batches.class_indices
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    filenames=test_batches.filenames
    results=pd.DataFrame({"Filename":filenames,
                          "DenseNet121_Predictions":predictions})
    results.to_csv("test_prediction_densenet121.csv", index=False)



predict_vgg()
predict_resnet()
predict_inceptionv3()
predict_densenet121()