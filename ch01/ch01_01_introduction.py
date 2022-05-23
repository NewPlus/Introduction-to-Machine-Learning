import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import scipy as sp
import IPython
import sklearn
import mglearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris_dataset = load_iris()

print("iris_dataset의 키: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("타깃의 이름: {}".format(iris_dataset['target_names']))
print("특성의 이름: \n{}".format(iris_dataset['feature_names']))
print("data의 타입: {}".format(type(iris_dataset['data'])))
print("data의 타입: {}".format(iris_dataset['data'].shape))
print("data의 처음 다섯 행:\n{}".format(iris_dataset['data'][:5]))
print("target의 타입:", type(iris_dataset['target']))
print("target의 크기:", iris_dataset['target'].shape)
print("타깃:\n", iris_dataset['target'])

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train 크기:", X_train.shape)
print("y_train 크기:", y_train.shape)
print("X_test 크기:", X_test.shape)
print("y_test 크기:", y_test.shape)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()