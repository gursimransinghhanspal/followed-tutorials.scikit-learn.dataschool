import numpy as np
from sklearn import metrics

# conventional way to import pandas
import pandas as pd

"""
Use pandas to read dataframes.
"""

# read CSV file directly from a URL and save the results
data = pd.read_csv('../datasets/Advertising/Advertising.csv', index_col = 0)
# display the first 5 rows
print("data.head():")
print(data.head())
print()
# display the last 5 rows
print("data.tail():")
print(data.tail())
print()

# check the shape of the DataFrame (rows, columns)
print("data.shape = " + str(data.shape))
print()

"""
Use seaborn to visualize the data.
"""

# conventional way to import seaborn
import matplotlib.pyplot as plt
import seaborn as sns
# visualize the relationship between the features and the response using scatterplots
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7, kind='reg')
## plot doesnt appear without this on macOS
# plt.show()

"""
Organize data in X and y.
"""

'''
# create a Python list of feature names
feature_cols = ['TV', 'Radio', 'Newspaper']
# use the list to select a subset of the original DataFrame
X = data[feature_cols]
'''
# equivalent command to do this in one line
X = data[['TV', 'Radio', 'Newspaper']]

# check the type and shape of X
print("type(X) = " + str(type(X)))					## pandas.core.frame.DataFrame
print("X.shape = " + str(X.shape))					## (200, 3)

'''
# select a Series from the DataFrame
y = data['Sales']
'''
# equivalent command that works if there are no spaces in the column name
y = data.Sales

# check the type and shape of y
print("type(y) = " + str(type(y)))					## pandas.core.series.Series
print("y.shape = " + str(y.shape))					## (200, )

"""
Split into training and testing datasets.
"""
'''
from sklearn.cross_validation import train_test_split

WARNING:
<...>/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
  "This module will be removed in 0.20.", DeprecationWarning)

SOLUTION:
use model_selection instead of cross_validation
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

# default split is 75% for training and 25% for testing
print("X_train.shape = " + str(X_train.shape))
print("y_train.shape = " + str(y_train.shape))
print("X_test.shape = " + str(X_test.shape))
print("y_test.shape = " + str(y_test.shape))
print()

"""
Use linear regression model.
"""

# import model
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()
print("linreg = " + str(linreg))
print()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)

# print the intercept and coefficients
print("linreg.intercept_ = " + str(linreg.intercept_))
print("linreg.coef_ = " + str(linreg.coef_))
print()
print(list(zip(['TV', 'Radio', 'Newspaper'], linreg.coef_)))
print()
# make predictions on the testing set
y_pred = linreg.predict(X_test)
print("linreg.predict(X_test):")
print(str(y_pred))
print()
print("RMSE(y_test, y_pred) = " + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))

print()
print("*" * 50)
print("*" * 50)
print()

"""
Model evaluation metrics for regression prediction
"""
# define true and predicted response values
trueArr = [100, 50, 30, 20]
predArr = [90, 50, 50, 30]

print("trueArr = " + str(trueArr))
print("predArr = " + str(predArr))

"""
1. Mean Absolute Error
"""
# calculate MAE by hand
print("MAE: " + str((10 + 0 + 20 + 10)/4.))
# calculate MAE using scikit-learn
print("MAE: metrics.mean_absolute_error(trueArr, predArr) = " + str(metrics.mean_absolute_error(trueArr, predArr)))

"""
2. Mean Squared Error
"""
# calculate MSE by hand
print("MSE: " + str((10**2 + 0**2 + 20**2 + 10**2)/4.))
# calculate MSE using scikit-learn
print("MSE: metrics.mean_squared_error(trueArr, predArr) = " + str(metrics.mean_squared_error(trueArr, predArr)))

"""
3. Root Mean Squared Error
"""
# calculate RMSE by hand
print("RMSE: " + str(np.sqrt((10**2 + 0**2 + 20**2 + 10**2)/4.)))
# calculate RMSE using scikit-learn
print("RMSE: np.sqrt(metrics.mean_squared_error(trueArr, predArr)) = " + str(np.sqrt(metrics.mean_squared_error(trueArr, predArr))))
print()

print()
print("*" * 50)
print("*" * 50)
print()

"""
Feature Selection:
Since newspaper has such a low coefficient for sales relation, we can try removing it from the training feature set.
"""

# use the list to select a subset of the original DataFrame
X = data[['TV', 'Radio']]
# select a Series from the DataFrame
y = data.Sales

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

# instantiate
linreg = LinearRegression()
print("linreg = " + str(linreg))
print()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)

# print the intercept and coefficients
print("linreg.intercept_ = " + str(linreg.intercept_))
print("linreg.coef_ = " + str(linreg.coef_))
print()
print(list(zip(['TV', 'Radio'], linreg.coef_)))
print()
# make predictions on the testing set
y_pred = linreg.predict(X_test)
print("linreg.predict(X_test):")
print(str(y_pred))
print()
print("RMSE(y_test, y_pred) = " + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))