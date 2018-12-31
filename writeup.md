# **Project 3: Traffic Sign Recognition** 
Author: Eduardo Escobar

## Writeup

This document presents the work developed for the 3rd project of the SDC Nanodegree.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./media/class_distribution.jpg "Visualization"
[image2]: ./media/traffic_signs.jpg "Traffic signs"
[image3]: ./media/augmentation.jpg "Data augmentation"
[image4]: ./media/gray_col.jpg "Gray scale"
[image5]: ./media/accuracy.jpg "Accuracy"
[image6]: ./media/loss.jpg "Loss"
[image7]: ./media/new_signs.jpg "New traffic signs"

[image6]: ./new_images/1.jpg "Traffic Sign 1"
[image7]: ./new_images/2.jpg "Traffic Sign 2"
[image8]: ./new_images/3.jpg "Traffic Sign 3"
[image9]: ./new_images/4.jpg "Traffic Sign 4"
[image10]: ./new_images/5.jpg "Traffic Sign 5"


### 1. Data Set Summary & Exploration

#### 1.1 Basic summary of the data set

I used 3 basic functionalities for numpy arrays: `len()`, `.shape` and `np.unique()` in order to obtain the sizes of the training, validation and test data.

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 1.2 Exploratory visualization of the dataset

Here is an exploratory visualization of the data set. It is a bar chart showing how many examples of each class are available at the training set. It can be seen that there is a high class disbalance, with some traffic signs appearing much more than others.

![alt text][image1]

Additionally, some random images were plotted with their respective labels, just to see how these images look. 

![alt text][image2]

### 2. Design and Test a Model Architecture

#### 2.1 Preprocessing the image data

In order to prepare the data for training the model, three kinds of pre-processing were used.

#### 2.1.1 Augment data

Since there are some classes heavily under-represented in the training set, three operations were used to create more images from the existing ones: rotation, random noise addition and horizontal flip.

![alt text][image3]

These operations were applied to the traffic signs with less examples than the average. The horizontal flip was only used with traffic signs that are horizontally symetrical.

This process of data augmentation was only applied to the training set, going from 34799 to 41603 train images.

#### 2.1.2 Normalize data

As suggested by the notebook, the data is normalized so it has zero mean and a standard deviation of one. In order to do these, the mean of each data set is substracted to all images, and then these images are divided by the standard deviation. 

#### 2.1.3 Convert images to gray scale

Finally, the images were converted to grayscale. This allows to create a second dataset,and then the color scale of the images becomes another hyperparameter.

![alt text][image4]

#### 2.2. Final model architecture

The lenet-5 architecture was used to train the model, since this architecture has proven to work well in previous lessons of the course. The only relevant addition that I did was to include dropout operations between the fully connected layers, in order to reduce the model's overfitting.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Fully connected		| input 400, output 120							|
| RELU					|												|
| Dropout				| keep probability: 0.7							|
| Fully connected		| input 120, output 84							|
| RELU					|												|
| Dropout				| keep probability: 0.7							|
|Fully connected		| input 84, output 43							|


#### 2.3 Training and validation of the model

To train the model, I mainly played with 4 hyper parameters:

* Number of epochs to train the model (final chosen value: 45)
* Barch size (final chose value: 128) 
* Color scale (RGB, gray) (final chosen value: RGB)
* Learning rate (final chosen value: 0.0008)

For a given set of these parameters, the training process was conducted. At each epoch, the values of accuracy and loss function were recorded and displayed for both the train and validation sets.

My final model results were:
* training set accuracy of 98.9%
* validation set accuracy of 94.5%
* test set accuracy of 92.6%

Here are the graphs for the loss and accuracy of the train and validation sets during the training epochs.

![alt text][image5]

![alt text][image6]


### 3. Test a Model on New Images

Here are five German traffic signs that I found on the web, with their predicted class and 5 top softmax probabilities:

![alt text][image7]


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy road      		| Beware of ice/snow							| 
| Priority road			| Priority road									|
| Right of way			| Right of way									|
| Turn right ahead 		| Turn right ahead      		 				|
| Slippery Road			| Speed limit (30km/h) 							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This does not compare favorably to the accuracy on the test set. This low value is due to the fact that the images used here may have some variations that were not present at the training set (for example, all these images were resized to 32x32 altering their original proportions). Also, 5 is a very low number of images to test the performance of the model property.

