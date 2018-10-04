import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense

images = []
steering_measurement = []

with open('../drives/drive1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = '../drives/drive1/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        steering_measurement.append(float(line[3]))

X_train = np.array(images)
y_train = np.array(steering_measurement)

model = Sequential()

model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')