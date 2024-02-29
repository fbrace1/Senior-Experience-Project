import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import shuffle

# Check if TensorFlow is using a GPU
print(tf.test.gpu_device_name())
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# exit()

import numpy as np
import pandas as pd
import cv2  
import os  
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow, subplots, show

# Load and display an image
image = imread(r'C:\Users\FBRAC\Projects\FredSeniorExperiment\Senior-Experience-Project\ARCard\Templates\Ace_D.jpg')
images = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
imshow(images[0])

# Data augmentation
data_generator = ImageDataGenerator(rotation_range=90, brightness_range=(0.5, 1.5), shear_range=15.0, zoom_range=[.3, .8])
data_generator.fit(images)
image_iterator = data_generator.flow(images)

# Display augmented images
# plt.figure(figsize=(16,16))
# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image_iterator.__next__()[0].astype('int'))
# plt.show()

# Load and preprocess all images
# Determine the number of images and their dimensions
num_images = len(os.listdir(r'C:\Users\FBRAC\Projects\FredSeniorExperiment\Senior-Experience-Project\ARCard\Templates')) * 750
image_shape = (180, 180, 1)

# Create an empty array with the fixed shape
data = np.empty((num_images, 2), dtype=object)

# Populate the data array
index = 0
for i, img in tqdm(enumerate(os.listdir(r'C:\Users\FBRAC\Projects\FredSeniorExperiment\Senior-Experience-Project\ARCard\Templates'))):
    label = i
    img = cv2.imread(os.path.join('C:\\Users\\FBRAC\\Projects\\FredSeniorExperiment\\Senior-Experience-Project\\ARCard\\Templates\\', img), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (180, 180))
    imgs = img.reshape((1, img.shape[0], img.shape[1], 1))
    data_generator = ImageDataGenerator(rotation_range=90, brightness_range=(0.5, 1.5), shear_range=15.0, zoom_range=[.3, .8])
    data_generator.fit(imgs)
    image_iterator = data_generator.flow(imgs)
    for x in range(750):
        img_transformed = next(image_iterator)[0].astype('float32') / 255
        img_transformed = cv2.resize(img_transformed, (180, 180))
        img_transformed = img_transformed[:, :, np.newaxis]  # Add a channel dimension

        if img_transformed.shape == image_shape:  # Check if the shape is correct
            data[index] = [img_transformed, label]
            index += 1
        else:
            print(f"Skipping image with shape {img_transformed.shape}")

# Save the array
np.save('C:/Users/FBRAC/Projects/DataSavedFromProgramRun/data.npy', data)

data = np.load('C:/Users/FBRAC/Projects/DataSavedFromProgramRun/data.npy', allow_pickle=True)


# Assuming 'data' is your dataset
shuffle(data)
# Split the data into training and testing sets
train = data[:35000]
test = data[35000:]

train_X, train_y = zip(*train)
test_X, test_y = zip(*test)

train_X = np.array(train_X)
train_y = np.array(train_y)
test_X = np.array(test_X)
test_y = np.array(test_y)

# Clear any previous session and set random seeds
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# Model configuration
epochs = 250
# epochs = 1

batch_size = 32
# batch_size = 1

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(train_X.shape[1], train_X.shape[2], train_X.shape[3])),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(55, activation='softmax')
])

model.summary()

# Set up a checkpoint callback
cp = tf.keras.callbacks.ModelCheckpoint(filepath="250epochs_conv.h5",
                               save_best_only=True,
                               verbose=0)

# Compile the model
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the model on the GPU
with tf.device('/gpu:0'):
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, 
                        validation_data=(test_X, test_y), callbacks=[cp]).history

# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
# from tensorflow.keras.regularizers import l2

# # Check if TensorFlow is using a GPU
# print(tf.test.gpu_device_name())
# print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# import numpy as np
# import cv2  
# import os  
# from tqdm import tqdm
# from matplotlib.pyplot import imread, imshow

# # Load and display an image
# image = imread(r'C:\Users\FBRAC\Projects\FredSeniorExperiment\Senior-Experience-Project\ARCard\ResizedTemplates\Ace_D.jpg')
# images = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# imshow(images[0])

# # Data augmentation
# data_generator = ImageDataGenerator(rotation_range=90, brightness_range=(0.5, 1.5), shear_range=15.0, zoom_range=[.3, .8])
# data_generator.fit(images)

# # Load and preprocess all images
# num_images = len(os.listdir(r'C:\Users\FBRAC\Projects\FredSeniorExperiment\Senior-Experience-Project\ARCard\ResizedTemplates')) * 750
# image_shape = (180, 180, 1)

# data = np.empty((num_images, 2), dtype=object)
# index = 0
# for i, img in tqdm(enumerate(os.listdir(r'C:\Users\FBRAC\Projects\FredSeniorExperiment\Senior-Experience-Project\ARCard\ResizedTemplates'))):
#     label = i
#     img = cv2.imread(os.path.join('C:\\Users\\FBRAC\\Projects\\FredSeniorExperiment\\Senior-Experience-Project\\ARCard\\ResizedTemplates\\', img), cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (180, 180))
#     imgs = img.reshape((1, img.shape[0], img.shape[1], 1))
#     data_generator.fit(imgs)
#     image_iterator = data_generator.flow(imgs)
#     for x in range(750):
#         img_transformed = next(image_iterator)[0].astype('float32') / 255
#         img_transformed = cv2.resize(img_transformed, (180, 180))
#         img_transformed = img_transformed[:, :, np.newaxis]

#         if img_transformed.shape == image_shape:
#             data[index] = [img_transformed, label]
#             index += 1

# # Save and load the array
# np.save('C:/Users/FBRAC/Projects/DataSavedFromProgramRun/data.npy', data)
# data = np.load('C:/Users/FBRAC/Projects/DataSavedFromProgramRun/data.npy', allow_pickle=True)

# # Split the data into training and testing sets
# train = data[:37000]
# test = data[37000:]
# train_X, train_y = zip(*train)
# test_X, test_y = zip(*test)
# train_X = np.array(train_X)
# train_y = np.array(train_y)
# test_X = np.array(test_X)
# test_y = np.array(test_y)

# # Clear any previous session and set random seeds
# tf.keras.backend.clear_session()
# np.random.seed(42)
# tf.random.set_seed(42)

# # Model configuration
# epochs = 4
# batch_size = 55

# # Define the model with L2 regularization
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(train_X.shape[1], train_X.shape[2], train_X.shape[3])),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0.8),  # Increased dropout rate
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(55, activation='softmax')
# ])

# model.summary()

# # Set up callbacks
# cp = tf.keras.callbacks.ModelCheckpoint(filepath="250epochs_conv.h5", save_best_only=True, verbose=0)
# es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# def scheduler(epoch, lr):
#     if epoch < 10:
#         return lr
#     else:
#         return lr * tf.math.exp(-0.1)

# lr_scheduler = LearningRateScheduler(scheduler)

# # Compile the model with a custom learning rate
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.05)
# model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# # Train the model on the GPU
# with tf.device('/gpu:0'):
#     history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, 
#                         validation_data=(test_X, test_y), callbacks=[cp, es, lr_scheduler]).history
