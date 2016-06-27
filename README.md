# Deep-Learning-Project

License to the author: Chun-Hao Liu

This project is to implement an End-to-End machine learning model to predict the housing price from the housing website Redfin (https://www.redfin.com). This project includes the following files to complete the whole machine learning process:

##1. redfin_2016-06-09-17-37-44_results.csv (an example for data input)

This is the Excel data sheet downloaded from Redfin website. It contains the current housing price and all its apartment features.

##2. Data_Preprocessing.py
This is a python script to do pre-processing for the Excel datasheet. It extracts the necessay input features and output and stores them numerically to Data_Preprocess.txt.

##3. main.lua
This is the main machine learning procedure file written in Lua and Torch. The End-to-End flow is as follows:
Data normalization -> Data separation (training set and test/validation set) -> Deep leanring model Creation -> Training -> Performance Evaluation -> Predicted data Storage

###3.1. Three deep learning model
multiple linear perceptron, convolutional neural network, and recurrent neural network.

###3.2. Two kinds of validation
Randomized training and testing, and k-fold cross-validation. 

###3.3 Two kinds of regularization
L1 and L2 norm.

##4. create_model.lua
Deep leanring model construction.

##5. train.lua
The whole training process is included in this file.

