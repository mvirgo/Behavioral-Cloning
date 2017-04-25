import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.utils import shuffle
import tensorflow as tf
import os
import matplotlib.image as mpimg
from scipy.misc import imresize

# Load all driving log data (3 images, angle, throttle, break, speed)
driving_data = pd.read_csv('driving_log.csv', sep=',', header = None)

# Make lists of only the center image names (to pull image data later) and steering angles
center_image_names = []
angles = []

for x in range(1, len(driving_data)):
    center_image = driving_data.get_value(x, 0)
    angle = driving_data.get_value(x, 3)
    center_image_names.append(center_image)
    angles.append(angle)

# Pull in all images
print('Loading images. This might take a bit.')
images = []
for i in center_image_names:
    image = mpimg.imread(i)
    image = imresize(image, (40, 80, 3))
    images.append(image)

# Make into numpy arrays
# The float part helps for visualizing at this point, if desired
images = np.array(images).astype(np.float)
angles = np.array(angles).astype(np.float)

print('All image loaded.')

# Change to typical training naming conventions
X_train, y_train = images, angles

# Shuffling first, then splitting into training and validation sets
X_train, y_train = shuffle(X_train, y_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=23)

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization

# Now for the actual neural network
print('Neural network initializing.')

batch_size = 100
epochs = 15
pool_size = (2, 2)
input_shape = X_train.shape[1:]

model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))

# Convolutional Layer 1 and Dropout
model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Conv Layer 2
model.add(Convolution2D(32, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))

# Conv Layer 3
model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))

# Conv Layer 4
model.add(Convolution2D(8, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))

# Pooling
model.add(MaxPooling2D(pool_size=pool_size))

# Flatten and Dropout
model.add(Flatten())
model.add(Dropout(0.5))

# Fully Connected Layer 1 and Dropout
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# FC Layer 2
model.add(Dense(64))
model.add(Activation('relu'))

# FC Layer 3
model.add(Dense(32))
model.add(Activation('relu'))

# Final FC Layer - just one output - steering angle
model.add(Dense(1))

# Compiling and training the model
model.compile(metrics=['mean_squared_error'], optimizer='Nadam', loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=2, validation_data=(X_val, y_val))

# Save model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

# Show summary of model
model.summary()
