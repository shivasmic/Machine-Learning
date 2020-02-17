#Polynomial Linear Regression
"""
Created on Thu Oct  3 19:17:05 2019

@author: SHIVAM
"""
#Since it is a small dataset hence we won't divide it into train and test datasets
#We need more and more information to make accurate predictions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Designing the linear regression model
from sklearn.linear_model import LinearRegression
lin_reg_1 = LinearRegression()
lin_reg_1.fit(X,y)

#Designing the polynomial regression model
#We use fit transform because we are creating the features
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualising linear regression model
plt.scatter(X,y)
plt.plot(X,lin_reg_1.predict(X),color = 'red')
plt.title('Linear Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualising the polynomial regression model
#To make it more generalize we use poly_reg.fit_transform()
plt.scatter(X,y)
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('Levels')
plt.ylabel('Salaries')
plt.show()

#Predicting salary using the linear regression model
z1 = np.array([6.5])
z1 = z1.reshape(-1,1)
y_pred1 = lin_reg_1.predict(z1)

#Predicting salary using the polynomial regression model
y_pred2 = lin_reg_2.predict(poly_reg.fit_transform(z1))