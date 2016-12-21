"""
Created on Thu Dec 15 10:40:11 2016

@author: harrymunro

Takes a dataframe as input from a csv file organised as a standard tabular spreadsheet.
Creates a numpy array from required columns and transforms data for implementation in sklearn.
Option to do linear regression (X on Y) for two variables.
Option to do linear regression with multiple Xs.
Option to do any other sklearn regression algorithm.
"""

from sklearn import linear_model
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import tree

filename = 'data.csv'
df = pd.read_csv(filename)
# optionally drop NaN values
df = df.dropna()

# set up paramaters
y = np.array(df['DWELL TIME'])
x1 = df['BOARDERS']
x2 = df['ALIGHTERS']
x3 = df['BOARDERS AND ALIGHTERS']
x4 = df['LINE ID']
x5 = df['DIRECTION CODE']

X = np.array([x3, x4, x5])
X = X.T

# lasoo linear regression
reg = linear_model.Lasso(alpha = 0.1)
reg.fit(X, y)
lassoo_score = reg.score(X, y)

# ordinary multiple linear regression
reg = linear_model.LinearRegression()
reg.fit(X, y)
ordinary_linear_score = reg.score(X, y)

# two variable linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(df['BOARDERS AND ALIGHTERS'],df['DWELL TIME'])

# decision tree regression (prone to overfitting!)
reg = tree.DecisionTreeRegressor()
reg = reg.fit(X, y)
tree_score = reg.score(X, y)
