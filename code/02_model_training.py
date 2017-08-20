# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()

"""
X is capital because it is a vector. y is a scalar.
"""

# store feature matrix in "X"
X = iris.data
# store response vector in "y"
y = iris.target

# print the shapes of X and y
print("X.shape = " + str(X.shape))								## (150, 4)
print("y.shape = " + str(y.shape))								## (150, )
print()

"""
scikit-learn 4 step modelling pattern:
1. Import the class you plan to use
2. Instantiate the estimator(model).
3. Fit the model with data. (aka "training")
4. Predict the response for a new value.
"""

"""
Using KNN Classifier, K=1
"""
from sklearn.neighbors import KNeighborsClassifier				## import

# if we don't specify the hyper-parameters now, they will be set to default values
knn = KNeighborsClassifier(n_neighbors = 1)						## instantiate
print("knn = " + str(knn))
print()

# this is an in-place operation
knn.fit(X, y)													## fit/train

# predict using "out of sample" data
result = knn.predict([[3, 5, 4, 2], [5, 4, 3, 2]])
print("knn.predict([[3, 5, 4, 2], [5, 4, 3, 2]] = " + str(result))		## predict
print()

"""
Using KNN Classifier, K=5
"""

# instantiate the model (using the value K=5)
knn = KNeighborsClassifier(n_neighbors = 5)
print("knn = " + str(knn))
print()

# fit the model with data
knn.fit(X, y)

# predict the response for new observations
result = knn.predict([[3, 5, 4, 2], [5, 4, 3, 2]])
print("knn.predict([[3, 5, 4, 2], [5, 4, 3, 2]] = " + str(result))		## predict
print()

"""
Using LogisticRegression
"""

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()
print("logreg = " + str(logreg))
print()

# fit the model with data
logreg.fit(X, y)

# predict the response for new observations
result = logreg.predict([[3, 5, 4, 2], [5, 4, 3, 2]])
print("logreg.predict([[3, 5, 4, 2], [5, 4, 3, 2]] = " + str(result))		## predict
print()