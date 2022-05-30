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

from sklearn.svm import SVC

X, y = mglearn.tools.make_handcrafted_dataset()                                                                  
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)                                                
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
# 데이터 포인트 그리기
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# 서포트 벡터
sv = svm.support_vectors_
# dual_coef_ 의 부호에 의해 서포트 벡터의 클래스 레이블이 결정됩니다
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.show()
# 일반적으로 결정 경계에 영향을 주는 데이터 포인터를 support vector라고 부른다
# 새로운 데이터 포인트와 각 서포트 벡터와의 거리 측정
# 분류 결정 by 서포트 벡터까지의 거리에 기반
# 각 서포트 벡터의 중요도는 훈련 과정에서 학습
# 가우시안 커널에 의해 데이터 포인트 사이의 거리 계산
# k_rbf(x_1, x_2) = exp(-gamma * |x_1-x_2|^2) (x_1, x_2 : 데이터 포인트, |x_1-x_2| : 유클리디안 거리, gamma : 가우시간 커널의 폭 제한 매개변수)
# gamma는 가우시안 커널 폭의 역수
# gamma가 작으면 넓은 영역, 큰 값이면 영향을 미치는 범위가 제한적 -> 반경 클수록 훈련 샘플의 영향 범위 증가
# C 매개 변수는 규제 매개 변수 -> 각 포인트의 중요도 제한

fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
        
axes[0, 0].legend(["class 0", "class 1", "class 0 support vector", "class 1 support vector"],ncol=4, loc=(.9, 1.2))
plt.show()

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # 위스콘신 유방암 데이터 셋

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

svc = SVC()
svc.fit(X_train, y_train)

print("훈련 세트 정확도: {:.2f}".format(svc.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(svc.score(X_test, y_test)))

plt.boxplot(X_train, manage_ticks=False)
plt.yscale("symlog")
plt.xlabel("feature list")
plt.ylabel("feature size")
plt.show()

# 훈련 세트에서 특성별 최솟값 계산
min_on_training = X_train.min(axis=0)
# 훈련 세트에서 특성별 (최댓값 - 최솟값) 범위 계산
range_on_training = (X_train - min_on_training).max(axis=0)

# 훈련 데이터에 최솟값을 빼고 범위로 나누면
# 각 특성에 대해 최솟값은 0 최댓값은 1 임
X_train_scaled = (X_train - min_on_training) / range_on_training
print("특성별 최솟값\n", X_train_scaled.min(axis=0))
print("특성별 최댓값\n", X_train_scaled.max(axis=0))

# 테스트 세트에도 같은 작업을 적용하지만
# 훈련 세트에서 계산한 최솟값과 범위를 사용합니다(자세한 내용은 3장에 있습니다)
X_test_scaled = (X_test - min_on_training) / range_on_training
svc = SVC()
svc.fit(X_train_scaled, y_train)

print("훈련 세트 정확도: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("테스트 세트 정확도: {:.3f}".format(svc.score(X_test_scaled, y_test)))

# C값을 늘린 경우
vc = SVC(C=20)
svc.fit(X_train_scaled, y_train)

print("훈련 세트 정확도: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("테스트 세트 정확도: {:.3f}".format(svc.score(X_test_scaled, y_test)))

# SVM은 데이터 특성이 적어도 복잡한 결정 경계를 만들 수 있음
# 저차원과 고차원의 데이터에 모두 잘 작동하지만 샘플이 너무 많으면 잘 작동하지 않음
# 데이터 전처리와 매개변 수 설정에 신경을 많이 써야 함.
# 랜덤 포레스트나 그래디언트 부스팅 같은 전처리가 거의 필요없는 트리 기반 모델을 많이 적용
# SVM은 분석하기도 어려움 -> 예측이 잘 되었는지, 모델 설명도 난해
# 모든 특성이 비슷한 단위이고 스케일이 비슷하면 SVM은 시도할 만 함
# 