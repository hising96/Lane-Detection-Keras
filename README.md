# Lane-Detection-Keras


**How To Use The Code**

We have uploaded two files in our code i.e. CNNmodel.py and lanes.py. **CNNmodel.py** creates a deep learning model using keras sequential layers. We have taken differnet parameter values for convilution, deconvolution,pooling and upsampling layers. This code trains and tests our model using Adam optimizer and MeanSquareError loss function. We have run the code fo 50 epochs. Batch Size is 130 and poolSize is (2,2). This code creates a **model.h5** file which can then be deployed on any new unseen video.

The next file **lanes.py** deploys the model.h5 file and draws lanes for any never before seen video. This file takes in an input video in any standard video format and then predicts lane lines and then prepares and output video in any video format desired by the user. 

**Pickling**

We have used serialization in python to pickle the datasets and its corresponding labels. We have include **topickle.py** file to pickle their datasets and corresponding labels to .p files.

**Concept**

Lane Detection as the name suggests identifies and marks the lanes on the road so as to assist vehicular movements. Lane Detection even has the capability of guiding blind drivers to a certain extent by helping them navigate in particular lanes and applying brakes when the lane marked area before the car falls less than a predefined value based on the size of the vehicle. One of the major challenges faced is one of different road and weather conditions which we have taken up and tried to solve. There is no effect of illumination changes and road surfaces in the final predicted lane output.

Deep Learning is a supervised learning paradigm that takes in labeled data-sets and develops the learning into a model. This model can then be used to predict the desired information for a completely new and unseen input i.e  input which was not used in the training process. Deep Learning has taken off in the recent years due to increase computation power. It is much accurate as compared to machine learning and does not require to write complete algorithms to get the task done. It consists of different layers and their associated functions which are chosen depending upon the problem at hand.

**Motivation**

Millions of lives are lost every year on roads due to unorganized traffic on roads. According to the data statistics by save Life responsibility of drivers is the top contributor to road crash deaths, accounting for 80.3 per cent deaths out of the total road crash fatalities in 2016. Out of the three vulnerabilities listed below , our model can efficiently solve the speeding and overtaking issues. Thus bringing down accidents on roads by 94.9 per cent. By calculating the marked area on road and observing if it is safe to change lanes or not we can bring down the number of accidents down by a great extent. The following statistics have been taken from Save Life official survey data.

According to this data if we perform similar calculations, our model is bound to avoid more than 90 per cent of such accidents which is truly revolutionary in its own sense. Coupled with other DAS~(driver assistance systems) like sign board detection and pedestrian detection our model has the capability to transform the way cars move on road.

**Methodology**

We have designed and trained a deep Convolutional Network from scratch for lane detection since a CNN based model is known to work best for image data. We have used many metrics values for hyper parameters and took the ones which gave the best result. The training is done on NVIDIA-DGX1. A deep learning approach has been used.

We have taken several layers in our code from the convolution, deconvolution, pooling and upsampling function provided in the sequential module of keras. We have laid all these layers one after the other in our model. We start with the convolutional layer where we have 8 filters. Filters determine the dimensionality of the output space.Thus our first layer has 8 dimensioned output. We have a kernel size of (3,3) which gives the length of 1D convolution window provided by the layer. Stride provides the stride length of layer which is (1,1) for the first layer. Padding is done on the input layer so that the output has the same length as the input. Rectified Linear Unit is used as activation function for this layer. 

**Results and Analysis**

We have taken a deep learning route in which we have laid out several keras sequential layers and tried to build a model using the same. Training and testing is done on the model and features are extracted from the road images. We have laid out several convolution, de-convolution and pooling layers and developed a FCN based architecture which is known to work best on image data. We have achieved an overall accuracy of 96.34 per cent.

Our model is a part of ADAS (automatic driver assistance systems) that can very easily help drivers to be safe and keep other safe too. Our model can be built with other models like pedestrian detection, signboard analysis and traffic recognition to built a robust navigation systems. Ride hailing services can deploy our model in its core state or coupled with others to ensure safety for drivers as well as passengers. 

**Conclusion And Future Work**

One of the greatest advances we would like to bring on-board is the lane detection in real time through mobile phone apps. It would really help real-time car drivers. When the demarcated area before the vehicle falls below a particular value depending on the size of the car either a lane change option or automatic braking is triggered. This would help avoid collisions to a great extent.\\
To avoid excessive lane changing safety messages are displayed for constant lane changes if there is sufficient area in front of the vehicle thus preventing rash driving.

During lane changing automatic lane changing headlights are turned on for other vehicles to notice and be safe.\\
We would also like to improve upon our model in terms of RNN as it is known to work best in case of sequence inputs. Lanes on roads have a sequence data form. RNN can work pretty well in this case of input.
