# Skin Cancer Dataset Preprocessing

# Import the libraries
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
import shutil

start_dir = 'start_dir/'

# Create a new directory for the images
base_dir = start_dir + 'base_dir'
os.mkdir(base_dir)
#
# Training file directory
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

# Validation file directory
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)

# Test file directory
test_dir = os.path.join(base_dir, 'test_dir')
os.mkdir(test_dir)
#
# Training file directory
train_dir = os.path.join(base_dir, 'train2_dir')
os.mkdir(train_dir)

# Validation file directory
val_dir = os.path.join(base_dir, 'val2_dir')
os.mkdir(val_dir)

# Test file directory
test_dir = os.path.join(base_dir, 'test2_dir')
os.mkdir(test_dir)
#
# Create new folders in the training directory for each of the classes
nv = os.path.join(train_dir, 'nv')
os.mkdir(nv)
mel = os.path.join(train_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(train_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(train_dir, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(train_dir, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(train_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(train_dir, 'df')
os.mkdir(df)

# Create new folders in the validation directory for each of the classes
nv = os.path.join(val_dir, 'nv')
os.mkdir(nv)
mel = os.path.join(val_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(val_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(val_dir, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(val_dir, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(val_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(val_dir, 'df')
os.mkdir(df)

# Create new folders in the test directory for each of the classes
nv = os.path.join(test_dir, 'nv')
os.mkdir(nv)
mel = os.path.join(test_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(test_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(test_dir, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(test_dir, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(test_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(test_dir, 'df')
os.mkdir(df)

# Read the metadata
df = pd.read_csv('train.csv')

# Display some information in the dataset
df.head()

# Set y as the labels
y = df['dx']

# Split the metadata into training and validation
df_train, df_val = train_test_split(df, test_size=0.2, random_state=None, stratify=y)


y2 = df_train['dx']

# Split the metadata into training and test
df_train, df_test = train_test_split(df_train, test_size=0.10, random_state=None, stratify=y2)

y3 = df_test['dx']

# Split the test set into training, validation and test
df_train2, df_test2 = train_test_split(df_test, test_size=0.2, random_state=None, stratify=y3)



# Print the shape of the training and validation split
print(df_train.shape)
print(df_val.shape)
print(df_test.shape)

# Find the number of values in the training and validation set
df_train['dx'].value_counts()
df_val['dx'].value_counts()
# df_test['dx'].value_counts()

df_train2['dx'].value_counts()
# df_val2['dx'].value_counts()
df_test2['dx'].value_counts()


# Transfer the images into folders
# Set the image id as the index
df.set_index('image_id', inplace=True)

# Get a list of images in each of the two folders
folder_1 = os.listdir('data')
# folder_2 = os.listdir('ham10000_images_part_2')

# Get a list of train and val images
train_list = list(df_train2['image_id'])
# val_list = list(df_val2['image_id'])
test_list = list(df_test2['image_id'])



# Transfer the training images
for image in train_list:
    fname = image + '.jpg'
    label = df.loc[image, 'dx']

    # source path to image
    src = os.path.join('/scratch/ai309/newdata', fname)
    # destination path to image
    dst = os.path.join(train_dir, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)


# Transfer the validation images
for image in val_list:

    fname = image + '.jpg'
    label = df.loc[image, 'dx']

    # source path to image
    src = os.path.join('/scratch/ai309/newdata', fname)
    # destination path to image
    dst = os.path.join(val_dir, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)

# Transfer the test images
for image in test_list:

    fname = image + '.jpg'
    label = df.loc[image, 'dx']

    # source path to image
    src = os.path.join('/scratch/ai309/newdata', fname)
    # destination path to image
    dst = os.path.join(test_dir, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)


# Augment the data
class_list = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

for item in class_list:

    # Create a temporary directory for the augmented images
    aug_dir = start_dir + 'aug_dir'
    os.mkdir(aug_dir)

    # Create a directory within the aug dir to store images of the same class
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)

    # Choose a class
    img_class = item

    # List all the images in the directory
    img_list = os.listdir(start_dir + 'base_dir/train_dir/' + img_class)

    # Copy images from the class train dir to the img_dir
    for fname in img_list:
        # source path to image
        src = os.path.join(start_dir + 'base_dir/train_dir/' + img_class, fname)
        # destination path to image
        dst = os.path.join(img_dir, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    # point to a dir containing the images and not to the images themselves
    path = aug_dir
    save_path = start_dir + 'base_dir/train_dir/' + img_class

    # Create a data generator to augment the images in real time
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.9,1.1),
        fill_mode='nearest')

    batch_size = 62

    aug_datagen = datagen.flow_from_directory(path,
                                              save_to_dir=save_path,
                                              save_format='jpg',
                                              target_size=(224, 224),
                                              batch_size=batch_size)

    # Generate the augmented images and add them to the training folders
    num_aug_images_wanted = 10000  # total number of images we want to have in each class
    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((num_aug_images_wanted - num_files) / batch_size))

    # run the generator and create about 10000 augmented images
    for i in range(0, num_batches):
        imgs, labels = next(aug_datagen)

    # delete temporary directory with the raw image files
    shutil.rmtree(aug_dir)


class_list = ['mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

for item in class_list:

    # Create a temporary directory for the augmented images
    aug_dir = start_dir + 'aug_dir'
    os.mkdir(aug_dir)

    # Create a directory within the aug dir to store images of the same class
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)

    # Choose a class
    img_class = item

    # List all the images in the directory
    img_list = os.listdir(start_dir + 'base_dir/val_dir/' + img_class)

    # Copy images from the class validation dir to the img_dir
    for fname in img_list:
        # source path to image
        src = os.path.join(start_dir + 'base_dir/val_dir/' + img_class, fname)
        # destination path to image
        dst = os.path.join(img_dir, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    # point to a dir containing the images and not to the images themselves
    path = aug_dir
    save_path = start_dir + 'base_dir/val_dir/' + img_class

    # Create a data generator to augment the images in real time
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.9,1.1),
        fill_mode='nearest')

    batch_size = 62

    aug_datagen = datagen.flow_from_directory(path,
                                              save_to_dir=save_path,
                                              save_format='jpg',
                                              target_size=(224, 224),
                                              batch_size=batch_size)

    # Generate the augmented images and add them to the training folders
    num_aug_images_wanted = 1341  # total number of images we want to have in each class
    num_files = len(os.listdir(img_dir))
    print("num of " + str(item) + " : " + str(num_files))
    num_batches = int(np.ceil((num_aug_images_wanted - num_files) / batch_size))
    print("num of batches:" + str(num_batches))

    # run the generator and create about 1341 augmented images
    for i in range(0, num_batches):
        imgs, labels = next(aug_datagen)

    # delete temporary directory with the raw image files
    shutil.rmtree(aug_dir)

    print("aug dir deleted")



