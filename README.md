# Behavioral-Cloning
##Udacity SDC Nanodegree Project 3

In this project, using image data gathered from a simulator, I trained a neural network to be able to drive the simulated car on its own around a track.

###Data Gathering

I gathered training data using Udacity's simulator, and chose to gather data from both training tracks. My model ended up being able to perform well enough to make it fully through both tracks.

After looking at some helpful tips online, I decided to go out and get a joystick to help with gathering data. The joystick helps to increase the amount of data outside of a zero degree steering angle - using a keyboard is lots of just momentary changes in angle, while the joystick helped to maintain steering angles around curves. Although my final data was still heavily centered around a zero angle, the full distribution was less than half at zero with all remaining angles making up greater than 60% of the total data. I also had a good distribution on both negative and positive steering angles (i.e. right and left turns).

![Histogram](https://github.com/mvirgo/Behavioral-Cloning/blob/master/Image_Histogram.png "Histogram of the steering angle distribution")

Another strategy I read was to make sure to add in lots of recovery data, whereby I start recording from a spot where the car would otherwise soon drift off the track, and therefore train it on the proper angle to recover with. I ran a few laps doing this, after only one or two of trying to stay in the middle of the lane. All in all, I gathered over 10,000 images from the center camera (there is also a left and right camera) to use for training.

![Recovery_left](https://github.com/mvirgo/Behavioral-Cloning/blob/master/Recovery_from_left.jpg "Recovering from the left")

*Gathering data to recover from the left*

![Recovery_right](https://github.com/mvirgo/Behavioral-Cloning/blob/master/Recovery_from_right.jpg "Recovering from the right")

*Gathering data to recover from the right*

![Straight](https://github.com/mvirgo/Behavioral-Cloning/blob/master/Good_driving.jpg "Driving straight")

*An image from driving mainly in the middle of the road*

This final total was after a few previous failures (some related to the data, some not, as mentioned below). A big difficulty I ran into was a sharp right turn in front of the lake - the original data I gathered kept sending me into the lake, as the car refused to take a sharp enough turn. However, after gathering another thousand or so images with sharp corrections, the training finally succeeded.

![My Nemesis](https://github.com/mvirgo/Behavioral-Cloning/blob/master/Nemesis.jpg "My Nemesis")

*I had a lot of trouble with this turn.*

As far as processing the images went, I resized them down to 25% size (instead of 160x320 I made them 40x80). My first attempt at using autonomous mode failed because of this - I had to update drive.py to include this resizing. Without resizing, the car did not move.

I also normalized the data at the start of my neural network using a batch normalization layer (which meant the autonomous mode automatically got normalized as it was run through the network as well).

Both loading and resizing the data (especially when getting to tens of thousands of images) can take awhile, so although my final model does not include it, I saved the images (and steering angles) into pickle files and reloaded as necessary to train my model to speed up the process from start to finish.

###Neural Network Architecture

Given that convolutional neural networks (CNNs) are typically good at identifying important features in images, I decided to do a CNN for my neural network. I started with a network similar to what I had done in a previous lab with Keras. This previous one had used 2 convolutional layers, 1 pooling layer, dropout, flattening, and two more fully connected layers (with dropout in between). I had used relu for all activations until the final one, where I used softmax.

Here I ran into another problem. When I entered autonomous mode, the car held a steady angle for a right turn, and ran straight off the track. I was a bit confused at first, before realizing I needed to remove the softmax activation from the end.

As noted above, I added in batch normalization to my network architecture, so that I did not have to do that separately on the training data I gathered and the test set from the simulator. That was the first layer after beginning my sequential model (before an convolutional layers).

From here, I deepened the network a bit to include more layers. The final one ended up with four convolutional layers (then the same pooling and flattening as mentioned above), followed by four fully connected layers. Note that the final fully connected layer needs to have an output of one, since it needs just one steering angle output. This architecture was what I found to arrive at the best training and validation scores.

I ended up placing dropout around some of the connections with a higher number of parameters (see parameter numbers by layer in my model summary below). After initially using dropout of 0.2 for all convolutional layers down through pooling, and 0.5 for all fully connected layers, I found that only putting them in these spots allowed for much further convergence for the model while still preventing any extreme overfitting.

The convolutional layers have filters totalling 64, 32, 16, and 8 in descending layers. The fully connected layers have output sizes of 128, 64, 32, and 1.

I used the validation scores in order to play around with some of the parameters. The different parameters I attempted were:

Pooling Size: (2,2) and (3,3)
Batch Size: 50, 100, 150
Kernel Size (convolutional layers): 2,2, 3,3, and 4,4
Epochs: 10, 15, 20
Valid vs. Same padding
Adam vs. Nadam optimizers
Number of Convolutional Filters (Half or double those mentioned above)
Output size of Fully Connected Layers (Half or double those mentioned above)

As can be seen in the model.py file, I went with a pooling size of (2,2), batch size of 100, kernel size of 3,3, 15 epochs, valid padding, the Nadam optimizer, and the layers mentioned already above. These produced the best validation results.

As far as the loss went, I used mean squared error, as the problem of steering angle is akin to a regression problem. I also had it show me the mean squared error as the metric spit out when running the program so that I could see if it was doing well on the real problem at hand.

The summary of my model is shown below.

![Summary](https://github.com/mvirgo/Behavioral-Cloning/blob/master/Model_Summary.png "Model Summary")

###Overall

This project was a challenging one, but it was incredibly rewarding to watch the simulated car autonomously drive its way around both tracks the whole way around! 
