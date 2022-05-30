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

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # 위스콘신 유방암 데이터 셋

from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
# 기본 값 : 깊이3, 트리100개, 학습률0.1
gbrt.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(gbrt.score(X_test, y_test)))

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
# 과적합 방지를 위해 최대 깊이 줄이기(깊이1)
gbrt.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(gbrt.score(X_test, y_test)))

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("feature importances")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)
plot_feature_importances_cancer(gbrt)
plt.show()
# 특성 중요도 그래프가 랜덤 포레스트와 비슷한 분포
# 다만 그래디언트 부스팅은 일부 특성을 완전히 무시

gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
# 과적합 방지를 위해 학습률 줄이기(학습률0.01)
gbrt.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(gbrt.score(X_test, y_test)))
# 최대 깊이 줄이기 또는 학습률 줄이기 -> 모델의 복잡도 감소
# 최대 깊이 줄이기 -> 테스트 세트 성능 크게 향상
# 학습률 줄이기 -> 테스트 세트 성능 약간 향상
# 보통은 안정적인 랜덤 포레스트 먼저 적용 -> 예측 시간이 중요, 마지막 성능까지 필요한 경우 -> 그래디언트 부스팅 사용
# 단점 : 매개 변수를 잘 조절해야 함, 훈련시간이 길다, 희소한 고차원 데이터에서는 잘 작동 안함(트리의 특징)
# 중요 매개 변수 : n_estimators(트리의 개수 지정), learning_rate(이진 트리의 오차 보정 정도 조절)
# n_estimators가 클수록 모델이 복잡, 과대적합 가능성 증가
# learning_rate를 낮추면 더 많은 트리가 필요(추가해야 함)