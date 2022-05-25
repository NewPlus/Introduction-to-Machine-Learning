import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import scipy as sp
import IPython
import sklearn
import mglearn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# 경고 메시지용

mglearn.plots.plot_knn_classification(n_neighbors=1)
# 가장 가까운 데이터 포인트를 연결(n_neighbors=1)
mglearn.plots.plot_knn_classification(n_neighbors=3)
# 가장 가까운 데이터 포인트 세 개를 연결(n_neighbors=3)
plt.show()

from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 데이터를 train_set과 test_set으로 나눔
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
# 가장 가까운 데이터 포인트 세 개를 연결(n_neighbors=3)
clf.fit(X_train, y_train)
# 모델 학습 : 예측할 때, 이웃을 찾을 수 있도록 데이터 저장

print("테스트 세트 예측:", clf.predict(X_test))
print("테스트 세트 정확도: {:.2f}".format(clf.score(X_test, y_test)))