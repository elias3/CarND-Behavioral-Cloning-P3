import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D

images = []
steering_measurement = []
correction = 0.2
path_to_images = '../drives/drive1/IMG/'
with open('../drives/drive1/driving_log.csv') as csvfile:
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

augmented_images = []
augmented_measurement = []

for image, measurement in zip(images,steering_measurement):
    augmented_images.append(image)
    augmented_measurement.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurement.append(-1.0*measurement)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurement)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

# model.add(Convolution2D(6,(5,5),strides =(2,2),activation="relu"))

model.add(Convolution2D(6,(5,5),activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,(5,5),activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')