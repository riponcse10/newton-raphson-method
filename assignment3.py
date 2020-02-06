#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:20:33 2018

@author: aminulhoque
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv('data.csv', na_values = '?')
X = dataset.iloc[:, 1:10].values
dfX = pd.DataFrame(X)
Y = dataset.iloc[:, 10].values
dfY = pd.DataFrame(Y)

#Filling the missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 5:6])
X[:, 5:6] = imputer.transform(X[:, 5:6])
dfX = pd.DataFrame(X)

#Changing the output value 2->0, 4->1
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = 'all')
Y = Y.reshape(-1,1)
Y = onehotencoder.fit_transform(Y).toarray()
dfY = pd.DataFrame(Y)
Y = dfY.drop(axis = 1, labels = 0)
dfY = pd.DataFrame(Y)

#sigmoid function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
    

seed = 42

#ten fold validation. 
ten_fold = 10;
errors = np.zeros((ten_fold, 1))
classification_accuracy = np.zeros((ten_fold, 1))
for sample in range(0, ten_fold):
    
    #Splitting to training and test data
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.2, random_state = seed)

    #Add intercept term to train_X
    intercept_X = np.array(np.ones((train_X.shape[0], 1)))
    train_X = np.column_stack((intercept_X, train_X))
    
    intercept_X_test = np.array(np.ones((test_X.shape[0], 1)))
    test_X = np.column_stack((intercept_X_test, test_X))
    
    #number of elements
    m = train_Y.shape[0]
    
    theta = np.zeros((train_X.shape[1], 1)) #Initialize parameters
    
    # Newtons method
    for i in range(0, 15):
        
        z = train_X.dot(theta)
        h = sigmoid(z)
        
        
        # Calculate gradient.
        grad = (1/m) * (train_X.T.dot(h - train_Y))
        
        #Calculate hessian
        H = train_X.T.dot((h*(1-h)*(train_X)))/m
        
        tmp = (-1)*train_Y*np.log(h) - (1-train_Y)*np.log((1-h))
        
        #Calculating J
        J = np.sum(tmp)/m
        print(str(J) + ' for iteration ' + str(sample))
        
        theta = theta - np.linalg.solve(H, grad) #updating theta
    
    row_test = test_X.shape[0]

    output = np.zeros((row_test, 1))
    
    #Output for test data using theta
    for j in range(0, row_test):
        t = pd.DataFrame(test_X).iloc[j, :].values
        t = t.dot(theta)
        if sigmoid(t) >= 0.5:
            output[j,0] = 1  
        else:
            output[j,0] = 0  
        
        
    #calculating error
    error = np.zeros((row_test, 1))
    correct_classification = 0;
    incorrect_classification = 0;
    for j in range(0, row_test):
        currenttestrow = pd.DataFrame(test_Y).iloc[j, :].values
        error[j] = (output[j] - currenttestrow[0])**2
        if output[j] == currenttestrow[0]:
            correct_classification = correct_classification + 1;
        else:
            incorrect_classification = incorrect_classification + 1;
        
        
    final_error =  (1/2) * np.sum(error)
    errors[sample,0] = final_error
    cls_accrcy = correct_classification/row_test
    classification_accuracy[sample] = cls_accrcy
    seed = seed + 10
    

average_error = np.mean(errors)
average_accuracy = np.mean(classification_accuracy)
print('Average error: '+ str(average_error))
print('Average accuracy: ' + str(average_accuracy))

    
    
    
    
    
    
    
    
    
    
    
    
    
    