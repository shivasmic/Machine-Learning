#Random Forest Regression
"""
Created on Sun Oct  6 19:00:36 2019

@author: SHIVAM
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Designing the Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,y)

#Predicting the value
z = np.array([6.5])
z = z.reshape(-1,1)
y_pred = regressor.predict(z)


#Visualisiing the results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Random Forest Regression Model')
plt.xlabel('Position Levels')
plt.ylabel('Salaries')
plt.show()


