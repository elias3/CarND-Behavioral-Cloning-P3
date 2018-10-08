import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.utils import plot_model


images = []
steering_measurement = []
correction = 0.2

# Given a path to the drive data, i.e. a folder named IMG that contains the images
# And a csv file named driving_log.csv
# the function adds the images to the list 'images' and the measurments to the list
# 'steering_measurement'

def process_data(path):
    path_to_images = path + 'IMG/'
    with open(path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            filename_center = line[0].split('/')[-1]
            filename_left = line[1].split('/')[-1]
            filename_right = line[2].split('/')[-1]

            path_to_image_center = path_to_images + filename_center
            path_to_image_left = path_to_images + filename_left
            path_to_image_right = path_to_images + filename_right

            image_center = cv2.imread(path_to_image_center)
            image_left = cv2.imread(path_to_image_left)
            image_right = cv2.imread(path_to_image_right)

            steering_angle = float(line[3])

            images.append(image_center)
            steering_measurement.append(steering_angle)
            images.append(image_left)
            steering_measurement.append(steering_angle + correction)
            images.append(image_right)
            steering_measurement.append(steering_angle - correction)

process_data('../drives/drive1/') # track 1 - center
process_data('../drives/track1_back/') # track 1 - center backwards
process_data('../drives/recovery1/') # track 1 - center backwards

augmented_images = []
augmented_measurement = []

# for every image we augment it by adding its flipped counterpart
for image, measurement in zip(images,steering_measurement):
    augmented_images.append(image)
    augmented_measurement.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurement.append(-1.0*measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurement)

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5,input_shape=(160,320,3)))

# Leaving the interesting area only
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
history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

print(history.history.keys())

model.save('model.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('training.png', bbox_inches='tight')
plt.show()

