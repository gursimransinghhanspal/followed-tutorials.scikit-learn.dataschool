"""
Introduction to scikit-learn using the famous IRIS dataset from 
the UCI Machine learning repository.
"""

"""
The IRIS dataset is built into scikit-learn by default.
"""

# import load_iris function from database module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
print("type(iris) type = " + str(type(iris)))					## sklearn.utils.Bunch
print()


# print the iris data
print("iris.data:")
print(iris.data)												## [[sl, sw, pl, pw, cat], ...]
print()

# print the names of the four features
print("iris.feature_names:") 
print(str(iris.feature_names))									## [sepal_length, sepal_width, ...]
print()

# print integers representing the species of each observation
print("iris.target:")
print(str(iris.target))											## [0, ..., 1, ..., 2, ...]
print()

# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
print("iris.target_names:")
print(str(iris.target_names))
print()

"""
1. The response and features(data) must be separate objects.
2. The response and features(data) must be numerical.
3. features(data) and response must be numPy arrays.
4. shape -> response.shape = data.shape[0]
"""

# check the types of the features and response
print("type(iris.data) = " + str(type(iris.data)))				## numpy.ndarray
print("type(iris.target) = " + str(type(iris.target)))			## numpy.ndarray
print()

# check the shape of the features (first dimension = number of observations, second dimensions = number of features)
print("iris.data.shape = " + str(iris.data.shape))				## (150, 4)

# check the shape of the response (single dimension matching the number of observations)
print("iris.target.shape = " + str(iris.target.shape))			## (150, )
print()