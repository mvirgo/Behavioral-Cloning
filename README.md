# Behavioral-Cloning
Udacity SDC Nanodegree Project 3

In this project, using image data gathered from a simulator, I trained a neural network to be able to drive the simulated car on its own around a track.

Data Gathering

I gathered training data using Udacity's simulator.

Although there are two tracks, I chose to only gather training data from the first track. This does have the side effect of limited usefulness on the second track, which has less distinct lane lines. I hope to one day come back and make an updated model using data on both tracks that can better generalize; however, for the time being, my model does a great job on track 1.

After looking at some helpful tips online, I decided to go out and get a joystick to help with gathering data. The joystick helps to increase the amount of data outside of a zero degree steering angle - using a keyboard is lots of just momentary changes in angle, while the joystick helped to maintain steering angles around curves. Although my final data was still heavily centered around a zero angle, the full distribution had about half at zero with all remaining angles making up the other half. I also had a good distribution on both negative and positive steering angles (i.e. right and left turns).

Another strategy I read was to make sure to add in lots of recovery data, whereby I start recording from a spot where the car would otherwise soon drift off the track, and therefore train it on the proper angle to recover with. I ran a few laps doing this, after only one or two of trying to stay in the middle of the lane. All in all, I gathered over 10,000 images from the center camera (there is also a left and right camera) to use for training.

This final total was after a few previous failures (some related to the data, some not, as mentioned below). A big difficulty I ran into was a sharp right turn in front of the lake - the original data I gathered kept sending me into the lake, as the car refused to take a sharp enough turn. However, after gathering another thousand or so images with sharp corrections, the training finally succeeded.

As far as processing the images went, I resized them down to 25% size (instead of 160x320 I made them 40x80). My first attempt at using autonomous mode failed because of this - I had to update drive.py to include this resizing. Without resizing, the car did not move.

I also normalized the data at the start of my neural network using a batch normalization layer (which meant the autonomous mode automatically got normalized as it was run through the network as well).

Both loading and resizing the data (especially when getting to tens of thousands of images) can take awhile, so although my final model does not include it, I saved the images (and steering angles) into pickle files and reloaded as necessary to train my model to speed up the process from start to finish.

Neural Network Architecture

Given that convolutional neural networks (CNNs) are typically good at identifying important features in images, I decided to do a CNN for my neural network. I started with a network similar to what I had done in a previous lab with Keras. This previous one had used 2 convolutional layers, 1 pooling layer, dropout, flattening, and two more fully connected layers (with dropout in between). I had used relu for all activations until the final one, where I used softmax.

Here I ran into another problem. When I entered autonomous mode, the car held a steady angle for a right turn, and ran straight off the track. I was a bit confused at first, before realizing I needed to remove the softmax activation from the end.

As noted above, I added in batch normalization to my network architecture, so that I did not have to do that separately on the training data I gathered and the test set from the simulator. That was the first layer after beginning my sequential model (before an convolutional layers).

From here, I deepened the network a bit to include more layers. The final one ended up with four convolutional layers (then the same pooling, dropout, and flattening as mentioned above), followed by four fully connected layers. Each of the fully connected layers included dropout, in order to prevent any potential overfitting. Note that the final fully connected layer needs to have an output of one, since it needs just one steering angle output. This architecture was what I found to arrive at the best training and validation scores.

The convolutional layers have filters totalling 64, 32, 16, and 8 in descending layers. The fully connected layers have output sizes of 128, 64, 32, and 1.

For pooling size, batch size, filter size (in the convolutional layers), epochs, and the nodes for the fully connected layers, I played around a bit to find a good spot for training and validation scores.

As far as the loss went, I used mean squared error, as the problem of steering angle is akin to a regression problem. I also had it show me the mean squared error as the metric spit out when running the program so that I could see if it was doing well on the real problem at hand.

I also used the Nadam optimizer over the Adam optimizer as that appeared to train better on my data.

Overall

This project was a challenging one, but it was incredibly rewarding to watch the simulated car autonomously drive its way around the entire track!
