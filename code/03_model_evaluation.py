# read in the iris data
from sklearn.datasets import load_iris
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target

"""
Procedure #1
Train and Test on the entire dataset.
"""

"""
Logistic Regression
"""

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()
print("logreg = " + str(logreg))
print()

# fit the model with data
logreg.fit(X, y)

# predict the response values for the observations in X
y_prediction = logreg.predict(X)							## len(y_prediction) = len(y)
print("logreg.predict(X):")
print(y_prediction)
print()

# compute classification accuracy for the logistic regression model
from sklearn import metrics

# called "training accuracy" when we train and test on the same data.
accuracy_score = metrics.accuracy_score(y, y_prediction)
print("training_accuracy [logreg] = " + str(accuracy_score))
print()

"""
KNN, K=5
"""

# import
from sklearn.neighbors import KNeighborsClassifier

# instantiate
knn = KNeighborsClassifier(n_neighbors = 5)
print("knn = " + str(knn))
print()

# fit
knn.fit(X, y)

# predict
y_prediction = knn.predict(X)							## len(y_prediction) = len(y)
print("knn.predict(X):")
print(y_prediction)
print()

# compute classification accuracy
accuracy_score = metrics.accuracy_score(y, y_prediction)
print("training_accuracy [knn, 5] = " + str(accuracy_score))
print()

"""
KNN, K=1
"""

# instantiate
knn = KNeighborsClassifier(n_neighbors = 1)
print("knn = " + str(knn))
print()

# fit
knn.fit(X, y)

# predict
y_prediction = knn.predict(X)							## len(y_prediction) = len(y)
print("knn.predict(X):")
print(y_prediction)
print()

# compute classification accuracy
accuracy_score = metrics.accuracy_score(y, y_prediction)
print("training_accuracy [knn, 1] = " + str(accuracy_score))
print()

"""
Procedure #2
Train/Test split.
"""

# STEP 1: split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)		# omitting random_state

# print the shapes of the new X objects
print("X_train.shape = " + str(X_train.shape))
print("X_test.shape = " + str(X_test.shape))
# print the shapes of the new y objects
print("y_train.shape = " + str(y_train.shape))
print("y_test.shape = " + str(y_test.shape))
print()

"""
Logistic Regression
"""

# STEP 2: train the model on the training set
logreg = LogisticRegression()
print("logreg = " + str(logreg))
print()

# fit the model with data
logreg.fit(X_train, y_train)

# predict the response values for the observations in X
y_prediction = logreg.predict(X_test)							## len(y_prediction) = len(y)
print("logreg.predict(X_test):")
print(y_prediction)
print()

# called "train test accuracy" when we train and test on the same data.
accuracy_score = metrics.accuracy_score(y_test, y_prediction)
print("traintest_accuracy [logreg] = " + str(accuracy_score))
print()

"""
KNN, K=5
"""

knn = KNeighborsClassifier(n_neighbors = 5)
print("knn = " + str(knn))
print()

# fit
knn.fit(X_train, y_train)

# predict
y_prediction = knn.predict(X_test)							## len(y_prediction) = len(y)
print("knn.predict(X_test):")
print(y_prediction)
print()

# calculate accuracy
accuracy_score = metrics.accuracy_score(y_test, y_prediction)
print("traintest_accuracy [knn, 5] = " + str(accuracy_score))
print()

"""
KNN, K=1
"""

knn = KNeighborsClassifier(n_neighbors = 1)
print("knn = " + str(knn))
print()

# fit
knn.fit(X_train, y_train)

# predict
y_prediction = knn.predict(X_test)							## len(y_prediction) = len(y)
print("knn.predict(X_test):")
print(y_prediction)
print()

# calculate accuracy
accuracy_score = metrics.accuracy_score(y_test, y_prediction)
print("traintest_accuracy [knn, 1] = " + str(accuracy_score))
print()

"""
Try improving on train test split.
Move through K=1 to K=25 to find the best value for K.
"""

k_range = list(range(1, 26))
knn_scores = []
for k in k_range:
	knn = KNeighborsClassifier(n_neighbors = k)
	knn.fit(X_train, y_train)
	y_prediction = knn.predict(X_test)
	knn_scores.append(metrics.accuracy_score(y_test, y_prediction))

'''
# import Matplotlib (scientific plotting library)
## if using virtualenv ie python is not installed as a framework.

import matplotlib												## macOS specific fix

## macOS specific fix for matplotlib
from sys import platform as _platform_name_						## macOS specific fix
if _platform_name_ == "darwin":									## macOS specific fix
	matplotlib.use('TkAgg')										## macOS specific fix

import matplotlib.pyplot as plt
del matplotlib													## macOS specific fix
'''

import matplotlib.pyplot as plt

# plot the relationship between K and testing accuracy
plt.plot(k_range, knn_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()