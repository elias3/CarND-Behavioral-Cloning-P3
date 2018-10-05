import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D

images = []
steering_measurement = []

with open('../drives/drive1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = '../drives/drive1/IMG/' + filename
        image = cv2.imread(current_path)
        steering_angle = float(line[3])

        images.append(image)
        steering_measurement.append(steering_angle)
        images.append(cv2.flip(image,1))
        steering_measurement.append(-1.0*steering_angle)


X_train = np.array(images)
y_train = np.array(steering_measurement)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
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