import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

import sklearn

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from random import shuffle

# images = []
# steering_measurement = []


samples = []
def add_samples(path_to_drive):
    with open(path_to_drive+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

add_samples('../drives/drive1/') # track 1 - center
add_samples('../drives/track1_back/') # track 1 - center backwards
add_samples('../drives/recovery1/') # track 1 - center backwards

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def add_image(images, angles, image, angle):
    images.append(image)
    angles.append(angle)
    images.append(cv2.flip(image,1))
    angles.append(-1.0*angle)

def generator(samples, batch_size=32):
    correction = 0.2
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                filename_center = batch_sample[0]
                filename_left = batch_sample[1]
                filename_right = batch_sample[2]

                image_center = cv2.imread(filename_center)
                image_left = cv2.imread(filename_left)
                image_right = cv2.imread(filename_right)

                angle = float(batch_sample[3])

                add_image(images, angles, image_center, angle)
                add_image(images, angles, image_left, angle + correction)
                add_image(images, angles, image_right, angle - correction)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))

model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dropout(0.05))
model.add(Dense(10))
model.add(Dropout(0.05))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
plot_model(model, to_file='model.png', show_shapes=True)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_samples),
    validation_data=validation_generator,
    validation_steps=len(validation_samples),
    epochs=3, verbose = 1)

model.save('model.h5')

print(history.history.keys())

# Plot training & validation accuracy values

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('training.png', bbox_inches='tight')
plt.show()

