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
from sklearn.linear_model import LogisticRegression

# 유방암 데이터 셋
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# c=1
logreg = LogisticRegression(max_iter=5000).fit(X_train, y_train)
print("C가 기본값 1인 훈련 세트 점수: {:.3f}".format(logreg.score(X_train, y_train)))
print("C가 기본값 1인 테스트 세트 점수: {:.3f}".format(logreg.score(X_test, y_test)))
# train과 test의 score가 비슷하므로 과소적합

# c=100
logreg100 = LogisticRegression(C=100, max_iter=5000).fit(X_train, y_train)
print("C=100 훈련 세트 점수: {:.3f}".format(logreg100.score(X_train, y_train)))
print("C=100 테스트 세트 점수: {:.3f}".format(logreg100.score(X_test, y_test)))
# train과 test의 score가 증가 -> 복잡도가 증가할수록 성능이 좋아짐

# c=0.01
logreg001 = LogisticRegression(C=0.01, max_iter=5000).fit(X_train, y_train)
print("C=0.01 훈련 세트 점수: {:.3f}".format(logreg001.score(X_train, y_train)))
print("C=0.01 테스트 세트 점수: {:.3f}".format(logreg001.score(X_test, y_test)))
# train과 test의 score가 더 감소 -> 왼쪽으로 더 이동 -> 정확도 더 감소

plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-5, 5)
plt.xlabel("feature")
plt.ylabel("coef size")
plt.legend()
plt.show()

# L1 규제를 사용한 Logistic Regression
for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(solver='liblinear', C=C, penalty="l1", max_iter=1000).fit(X_train, y_train)
    print("C={:.3f} 인 l1 로지스틱 회귀의 훈련 정확도: {:.2f}".format(C, lr_l1.score(X_train, y_train)))
    print("C={:.3f} 인 l1 로지스틱 회귀의 테스트 정확도: {:.2f}".format(C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.xlabel("특성")
plt.ylabel("계수 크기")

plt.ylim(-5, 5)
plt.legend(loc=3)
plt.show()