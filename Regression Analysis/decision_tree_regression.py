#Decision Tree Regression Model
"""
Created on Sat Oct  5 14:18:30 2019

@author: SHIVAM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Designing the Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

#Creating the test values
z = np.array([6.5])
z = z.reshape(-1,1)
y_pred = regressor.predict(z)

#Plotting the Results
#X_grid for better visualisation and smoother curve
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Decision Tree Regressor')
plt.xlabel('levels')
plt.ylabel('Salaries')
plt.show()