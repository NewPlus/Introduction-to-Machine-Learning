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
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]]) # 야생에서 [꽃받침의 길이, 폭, 꽃잎 길이, 폭]
prediction = knn.predict(X_new)
print("예측:", prediction)
print("예측한 타깃의 이름:", iris_dataset['target_names'][prediction])

y_pred = knn.predict(X_test)
print("테스트 세트에 대한 예측값:\n{}".format(y_pred))
print("테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred == y_test)))
print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test, y_test)))