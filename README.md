# Lane-Detection-Keras


**How To Use The Code**

We have uploaded two files in our code i.e. CNNmodel.py and lanes.py. **CNNmodel.py** creates a deep learning model using keras sequential layers. We have taken differnet parameter values for convilution, deconvolution,pooling and upsampling layers. This code trains and tests our model using Adam optimizer and MeanSquareError loss function. We have run the code fo 50 epochs. Batch Size is 130 and poolSize is (2,2). This code creates a **model.h5** file which can then be deployed on any new unseen video.

The next file **lanes.py** deploys the model.h5 file and draws lanes for any never before seen video. This file takes in an input video in any standard video format and then predicts lane lines and then prepares and output video in any video format desired by the user. 

**Pickling**

We have used serialization in python to pickle the datasets and its corresponding labels. We have include **topickle.py** file to pickle their datasets and corresponding labels to .p files.

**Concept**

Lane Detection as the name suggests identifies and marks the lanes on the road so as to assist vehicular movements. Lane Detection even has the capability of guiding blind drivers to a certain extent by helping them navigate in particular lanes and applying brakes when the lane marked area before the car falls less than a predefined value based on the size of the vehicle. One of the major challenges faced is one of different road and weather conditions which we have taken up and tried to solve. There is no effect of illumination changes and road surfaces in the final predicted lane output.

Deep Learning is a supervised learning paradigm that takes in labeled data-sets and develops the learning into a model. This model can then be used to predict the desired information for a completely new and unseen input i.e  input which was not used in the training process. Deep Learning has taken off in the recent years due to increase computation power. It is much accurate as compared to machine learning and does not require to write complete algorithms to get the task done. It consists of different layers and their associated functions which are chosen depending upon the problem at hand.

