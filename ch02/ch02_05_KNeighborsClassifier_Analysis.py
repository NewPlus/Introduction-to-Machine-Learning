import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import scipy as sp
import IPython
import sklearn
import mglearn

from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

# 2차원 데이터셋이므로 가능한 모든 테스트 포인트의 예측을 xy평면에 그릴 수 있음
# 각 데이터 포인트가 속한 영역을 색칠
# 각 영역으로 나뉘는 결정 경계(Decision boundary)
 
for n_neighbors, ax in zip([1, 3, 9], axes): # [1, 3, 9]은 각 n_neighbors의 값들
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    # 1, 3, 9일때 KNeighbors 분석을 하고 모델 학습
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    # xy평면에 나타냄
    ax.set_title("{} Neighbor".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()

# 이웃의 개수가 1에서 9로 갈수록 결정경계가 부드러워 짐.
# 이웃이 적으면 모델의 복잡도 증가, 많으면 복잡도 감소
# 훈련 데이터 전체 개수를 이웃의 수로 정하는 극단적인 경우
# 테스트 포인트에 대한 예측은 모두 같은 값이 된다.
