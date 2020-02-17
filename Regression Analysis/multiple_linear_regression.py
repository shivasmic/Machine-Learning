"""
Created on Wed Oct  2 13:09:21 2019

@author: SHIVAM
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset and converting it into the training and testing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Converting the last independent variabble into dummy variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Save the program from dummy variable trap = it reduces the redundent relationship = only n-1 dummy variables are required 
X = X[:, 1:] 

#Split the data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 0)

#Designing the Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test results 
y_pred = regressor.predict(X_test)

#Building the optimal regression model using the Backward Eleimination
#We are doing this to create b0 variable in our regression equation
#OLS means Ordinary Least Squares
import statsmodels.api as sm
Significance_Level = 0.05
X = np.append(arr = np.ones((50,1)), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()



