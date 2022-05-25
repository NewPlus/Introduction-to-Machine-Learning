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
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.legend(["class 0", "class 1", "class 2"])
plt.show()

# LinearSVC 분류기
linear_svm = LinearSVC().fit(X, y)
print("계수 배열의 크기: ", linear_svm.coef_.shape)
print("절편 배열의 크기: ", linear_svm.intercept_.shape)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.legend(['class 0', 'class 1', 'class 2', 'class 0 boundary', 'class 1 boundary', 'class 2 boundary'], loc=(1.01, 0.3))
plt.show()
# 각 boundary가 자신의 class를 제외한 나머지 class의 data-points를
# 나머지 영역에 두는 식으로 분류함
# 중앙의 삼각형 영역 = 분류 공식의 결과가 가장 높은 class = 가장 가까운 직선의 class

mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.legend(['class 0', 'class 1', 'class 2', 'class 0 boundary', 'class 1 boundary', 'class 2 boundary'], loc=(1.01, 0.3))
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.show()
# Linear model의 주요 매개변수
# Regression model : alpha, LinearSVC, LogisticRegression: C
# alpha가 클수록, C가 작을수록 모델이 단순해짐
# C, alpha의 값은 로그 스케일(10단위)로 정함
# L1, L2 규제를 선택해야 한다.
# if 중요한 특성이 별로 없으면 L1, else L2 규제
# 학습 속도, 예측이 빠르다
# 희소한 데이터 셋, 매우 큰 데이터 셋도 잘 작동
# 본 공식을 통해 예측의 과정을 잘 이해할 수 있음
# 단, 각 데이터 셋 특성들이 서로 깊게 연관된 경우 왜 그런지 명확하지 않음 