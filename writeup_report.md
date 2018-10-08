# **Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Behavioral Cloning Project
---

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


[center]: ./images/track1_center.jpg "Track 1 Center"
[center_back]: ./images/track1_center_back.jpg "Track 1 Center"
[normal]: ./images/normal.jpg "Normal"
[flipped]: ./images/flipped.jpg "Flipped"

[recovery1]: ./images/recovery1.jpg "Recovery 1"
[recovery2]: ./images/recovery2.jpg "Recovery 2"
[recovery3]: ./images/recovery3.jpg "Recovery 3"


[model]: ./model.png "Model"
[training]: ./training.png "Training"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model using a generator
* network.py containing the script to create and train the model without a generator
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md (this file) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py and network.py files contain the code for training and saving the convolution neural network. The files show the pipeline I used for training and validating the model, and they contain comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3 layers of convolutions that have 5x5 filter sizes and a stride of 2 followed by two layers of 3x3 filters without a stride and depths between 24 and 64 (model.py lines 72-76)

The model includes RELU layers to introduce nonlinearity (code line 72-76), and the data is normalized in the model using a Keras lambda layer (code line 69). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 81,83,85). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 91-92). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 88).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to rely on a common CNN that already solved this problem and tune it for our purposes.

My first step was to use a convolution neural network model similar to LeNet I thought this model might be appropriate because LeNet is a well known network that showed good results in classification problems. Afterwards I decided to adapt the model that was used in the paper "End to End Learning for Self-Driving Cars", 2016.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by adding drop-out layers.

Then I also added more data by adding more laps, backward driving and recovery driving.

The final step was to run the simulator to see how well the car was driving around track one. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
This can be shown in the following video:

[link to my video result](./video.mp4)

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) is the following:

![alt text][model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded two laps on track one using center lane driving in the opposite direction.

![alt text][center_back]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it is off track. These images show what a recovery looks like:

![alt text][recovery1]
![alt text][recovery2]
![alt text][recovery3]

To augment the data sat, I also flipped images and angles thinking that this would. For example, here is an image that has then been flipped:

![alt text][normal]
![alt text][flipped]

After the collection process, I had 51072 number of data points. I then preprocessed this data by cropping the borders that include non-relevant information and normalized the pixel values by dividing by 255.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the loss graph. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The final training loss graph looks like the following:

![alt text][training]