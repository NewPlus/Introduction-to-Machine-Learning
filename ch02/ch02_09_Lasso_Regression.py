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

# 라쏘(Lasso)
# 릿지처럼 라쏘도 계수를 0으로 만들려는 특성이 있음
# 다만 방식이 조금 다름(L1 규제)
# L1 규제의 결과로 정말로 어떤 계수는 0이 됨
# 모델에서 완전히 제외되는 특성이 생긴다는 뜻 -> 특성 선택(feature selection)이 자동

# 보스턴 주택 가격 데이터 셋
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.linear_model import Lasso

# alpha가 1인 Lasso
lasso = Lasso().fit(X_train, y_train)
print("Lasso 훈련 세트 점수: {:.2f}".format(lasso.score(X_train, y_train)))
print("Lasso 테스트 세트 점수: {:.2f}".format(lasso.score(X_test, y_test)))
print("Lasso가 사용한 특성의 개수:", np.sum(lasso.coef_ != 0))
# train, test 모두 score가 낮다

# alpha가 0.01인 Lasso(max_iter 기본 값을 증가시키지 않으면 늘리라는 경고가 뜸)
lasso001 = Lasso(alpha=0.01, max_iter=50000).fit(X_train, y_train)
print("Lasso0.01 훈련 세트 점수: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Lasso0.01 테스트 세트 점수: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Lasso0.01가 사용한 특성의 개수:", np.sum(lasso001.coef_ != 0))
# train, test 모두 score가 좋아짐

# alpha가 0.0001인 Lasso
lasso00001 = Lasso(alpha=0.0001, max_iter=50000).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("사용한 특성의 개수:", np.sum(lasso00001.coef_ != 0))
# train은 좋아졌지만 test는 낮아짐 -> 오버피팅

# 비교용 릿지0.1
from sklearn.linear_model import Ridge
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)


plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")

plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("coef List")
plt.ylabel("coef Size")
plt.show()