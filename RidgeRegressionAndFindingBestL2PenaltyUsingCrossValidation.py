# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:35:51 2019

@author: rajui
"""

#importing packages
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plot 

#This function is used to load CSV file from the 'data' directory 
#in the present working directly 
def loadCSV (fileName):
    scriptDirectory = os.path.dirname(__file__)
    dataDirectoryPath = "."
    dataDirectory = os.path.join(scriptDirectory, dataDirectoryPath)
    dataFilePath = os.path.join(dataDirectory, fileName)
    return pd.read_csv(dataFilePath)

#This funtion is used to preview the data in the given dataset
def previewData (dataSet):
    print(dataSet.head())
    print("\n")

#This function is used to check for missing values in a given dataSet
def checkForMissingValues (dataSet):
    print(dataSet.isnull().sum())
    print("\n")

#This function is used to check the statistics of a given dataSet
def getStatisticsOfData (dataSet):
    print("***** Datatype of each column in the data set: *****")
    dataSet.info()
    print("\n")
    print("***** Columns in the data set: *****")
    print(dataSet.columns.values)
    print("***** Details about the data set: *****")
    print(dataSet.describe())
    print("\n")
    print("***** Checking for any missing values in the data set: *****")
    checkForMissingValues(dataSet)
    print("\n")

#This funtion is used to handle the missing value in the features, in the 
#given examples
def handleMissingValues (feature):
    feature = np.array(feature).reshape(-1, 1)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
    imputer.fit(feature)
    feature_values = imputer.fit_transform(feature)
    return feature_values
    
#Define file names and call loadCSV to load the CSV files
dataFile = "Advertising.csv"
dataSet = loadCSV(dataFile)
dataSet.drop(['Unnamed: 0'], axis=1, inplace=True)

#Preview the dataSet and look at the statistics of the dataSet
#Check for any missing values 
#so that we will know whether to handle the missing values or not
print("** Preview the dataSet and look at the statistics of the dataSet **")
previewData(dataSet)
getStatisticsOfData(dataSet)

#In this example we will be performing the Ridge regression to compute the model
#that will be used to predict the sales given the information about how much 
#marketing was performed using different channels like TV, Radio and Newspaper

#We are dropping the sales column from dataset which is a label
features = dataSet.drop(['sales','newspaper','radio'], axis=1)
label = dataSet['sales'].values.reshape(-1,1)

#Splitting the data into Train and Test
from sklearn.model_selection import train_test_split 
featuresForTraining, featuresForTesting, labelForTraining, labelForTesting = train_test_split(features,label,test_size=0.25, random_state=0)

#This is the tuning parameter to balance the fit of data and 
#magnitude of coefficients
#Here we use multiple alpha values with which we perform the Ridge regression
#and select the best value for alpha using cross validation 
alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
parameters = {'alpha': alpha}

#Importing the GridSearchCV, which is a model evaluation tool that uses 
#cross-validation. It rely on an internal scoring strategy for evaluating 
#the quality of a model’s predictions
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

#Defining the Ridge regressor, which will be used to compute the model using
#Ridge regression
ridge = Ridge()
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(featuresForTraining, labelForTraining)
print ('Ridge Regression - Best L2 penalty identified using cross validation is: ', ridge_regressor.best_params_)
print ('Ridge Regression - Negative mean square error is: ', ridge_regressor.best_score_)

#Predicting the prices
predictionsFromRidgeRegression = ridge_regressor.predict(featuresForTesting)
#Calculating the Mean of the squared error
from sklearn.metrics import mean_squared_error
print ("Ridge Regression - Mean squared error: ", 
       mean_squared_error(labelForTesting, predictionsFromRidgeRegression))

#Finding out the accuracy of the model
from sklearn.metrics import r2_score
accuracyMeassure = r2_score(labelForTesting, predictionsFromRidgeRegression)
print ("Ridge Regression - Accuracy of model is {} %".format(accuracyMeassure*100))

#Visualizing the training Test Results 
#Scatter plot the training dataset
plot.scatter(featuresForTraining, labelForTraining, color= 'green', marker='+')
#Scatter plot the test dataset
plot.scatter(featuresForTesting, labelForTesting, color= 'blue', marker='+')
#Plot the regression line
plot.plot(featuresForTesting, predictionsFromRidgeRegression, color = 'red')
plot.title ("Visuals for Ridge Regression")
plot.xlabel("Marketing Via TV")
plot.ylabel("Sales")
plot.show()

#Comparing with linear regression
from sklearn.linear_model import LinearRegression

#Fitting simple linear regression to the Training Set
regressor = LinearRegression()
regressor.fit(featuresForTraining, labelForTraining)

#Predicting the prices
predictionFromLinearRegression = regressor.predict(featuresForTesting)

#The coefficients / the linear regression weights
print ('\nLinear Regression - Coefficients: ', regressor.coef_)
#Calculating the Mean of the squared error
from sklearn.metrics import mean_squared_error
print ("Linear Regression - Mean squared error: ", 
       mean_squared_error(labelForTesting, predictionFromLinearRegression))
#Finding out the accuracy of the model
from sklearn.metrics import r2_score
accuracyMeassure = r2_score(labelForTesting, predictionFromLinearRegression)
print ("Linear Regression - Accuracy of model is {} %".format(accuracyMeassure*100))

#Visualizing the training Test Results 
#Scatter plot the training dataset
plot.scatter(featuresForTraining, labelForTraining, color= 'green', marker='+')
#Scatter plot the test dataset
plot.scatter(featuresForTesting, labelForTesting, color= 'blue', marker='+')
#Plot the regression line
plot.plot(featuresForTesting, predictionFromLinearRegression, color = 'red')
plot.title ("Visuals for Linear Regression with single feature")
plot.xlabel("Marketing Via TV")
plot.ylabel("Sales")
plot.show()

