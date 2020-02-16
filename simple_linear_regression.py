"""
Created on Mon Sep 30 20:01:24 2019

@author: SHIVAM
"""

#Simple Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Splitting the data into train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Creating the simple linear regression model
from sklearn.linear_model import LinearRegression

#Our Model/machine is regressor 
regressor = LinearRegression()

#Fit means learning the correlation between x and y
regressor = regressor.fit(X_train, y_train)

#Predicting the values from our model
y_pred = regressor.predict(X_test)

#Data Visualization for Training set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Experience(Training) vs Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#Data Visualization for Test set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #same line has to be fitted into the test sets
plt.title('Experience(Test) vs Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
