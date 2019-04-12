# Deep U-Net for B-format Speech Separation with Gaussian-Kernel Fitting of Angular Features

******************************************************************************************
This code is implemented in Python, using [Keras](https://keras.io/) with [Tensorflow](https://www.tensorflow.org/) backend where "image_data_format": "channels_last" was set.


## Data preparation

First run codes in audioProcess to divide all the data to train and testing data, and some temporary log files TIMITtraniningSignalList.pkl and TIMITtestingSignalList.pkl will be generated. 


## Train

To train the source separation models, run TrainModel.py. You can choose using different features and DNN structures and loss functions .etc. 

## Test

Run TestModel.py

## Evaluation

Run Matlab codes in ./RIRs
