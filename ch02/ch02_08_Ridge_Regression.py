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

from sklearn.linear_model import Ridge

# 보스턴 주택 가격 데이터 셋
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 비교용 LinearRegression
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)

# 릿지 회귀(Ridge Regression)
ridge = Ridge().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge.score(X_test, y_test)))
# 추가 제약조건을 만족시키기 위해 가중치(w)를 선택
# 가능한 한 가중치의 절댓값을 작게 만든다 -> w의 모든 원소가 0에 가깝도록
# 모든 특성이 출력에 주는 영향을 최소한으로 만듦. -> 기울기를 작게
# 릿지 결과 과대적합이 감소

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("ridge10 훈련 세트 점수: {:.2f}".format(ridge10.score(X_train, y_train)))
print("ridge10 테스트 세트 점수: {:.2f}".format(ridge10.score(X_test, y_test)))
# ridge10의 경우 계수를 0에 더 가깝게 만듦 -> 훈련 성능 저하, 일반화 증가

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("ridge0.1 훈련 세트 점수: {:.2f}".format(ridge01.score(X_train, y_train)))
print("ridge0.1 테스트 세트 점수: {:.2f}".format(ridge01.score(X_test, y_test)))
# ridge01의 경우 계수에 대한 제약이 많이 풀림 -> 훈련, 테스트 성능 증가 -> 평범한 LinearRegression과 같아짐

plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("coef List")
plt.ylabel("coef Size")
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-25, 25)
plt.legend()
plt.show()

mglearn.plots.plot_ridge_n_samples()
plt.show()
# 훈련 데이터에서는 score(R^2)가 LinearRegression > Ridge이지만
# 테스트 데이터에서는 LinearRegression < Ridge (size < 500 일 때)
# 또한 데이터의 크기가 커질수록 LinearRegression과 Ridge가 성능이 비슷해지는 것을 알 수 있다.
# 데이터가 많아질수록 과대적합하기 어려워짐.
