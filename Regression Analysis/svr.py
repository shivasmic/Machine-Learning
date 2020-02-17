#Support Vector Regression
"""
Created on Sat Oct  5 13:00:37 2019

@author: SHIVAM
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset =  pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
y = y.reshape((len(y), 1))

#SVR only works with the scaler data as it is  a less commonly used model
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Designing the Support Vector Regression model
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

#Predicting the results from the SVR model
z = np.array([6.5])
z = z.reshape(-1,1)
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(z)))

#Visualising the results
plt.scatter(X,y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title('SVR Model')
plt.xlabel('Levels')
plt.ylabel('Salaries')
plt.show()