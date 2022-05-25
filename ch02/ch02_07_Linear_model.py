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
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# 경고 메시지용

mglearn.plots.plot_linear_regression_wave()
plt.show()
# 선형 함수를 이용하여 예측함.
# 회귀를 위한 선형 모델은 특성이 하나면 직선, 두 개면 평면이 된다
# 더 많은 특성에서는 초평면(hyperplane)이 된다.
# 데이터가 항상 선형적으로 완벽하게 떨어질 수는 없기 때문에
# 일부 상세 데이터를 잃는 것은 사실이나 특성이 많은 데이터 셋에서
# 선형 모델은 아주 훌륭한 성능을 낸다.
# 특히 훈련 데이터보다 특성이 더 많은 경우라면 어떤 타깃도 선형 함수로 훈련 데이터에 대해 완벽하게 모델링 가능하다.

from sklearn.linear_model import LinearRegression
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 선형 회귀(LinearRegression) 혹은 오차제곱법(OLS, ordinary least squares)
lr = LinearRegression().fit(X_train, y_train)
print("lr.coef_:", lr.coef_)
print("lr.intercept_:", lr.intercept_)
# 예측과 훈련 세트 속 타깃 y의 평균제곱오차(mean squared error)를
# 최소화하는 파라미터 w와 b를 찾는다.
# 차이 제곱 -> 더함 -> 샘플의 개수로 나눔
# 선형 회귀는 매개변수가 없지만 모델의 복잡도를 제어하기 어려움\
print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))
# 과소 적합이 남.

# Boston Housing Price Extended Dataset
X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))
# 훈련 세트의 점수는 높지만 테스트 세트의 점수가 낮음 -> 과적합이 남.
# 복잡도를 제어할 수 있는 모델이 필요함.