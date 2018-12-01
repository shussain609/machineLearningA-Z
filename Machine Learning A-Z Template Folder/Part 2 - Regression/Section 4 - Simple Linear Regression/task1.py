# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:37:26 2018

@author: shadab
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#taking care of missing data


#encodinng categorical data


#splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
#apply 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predict
y_pred = regressor.predict(x_test)
#plotting
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('linear regression model train set')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('linear regression model test set')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()