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

print("유방암 데이터의 특성별 최대값:\n", cancer.data.max(axis=0))

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print("훈련 세트 정확도: {:.2f}".format(mlp.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(mlp.score(X_test, y_test)))

# 데이터 셋의 모든 입력 특성을 평균 0, 분산 1로 전처리
# 훈련 세트 각 특성의 평균을 계산합니다
mean_on_train = X_train.mean(axis=0)
# 훈련 세트 각 특성의 표준 편차를 계산합니다
std_on_train = X_train.std(axis=0)

# 데이터에서 평균을 빼고 표준 편차로 나누면
# 평균 0, 표준 편차 1 인 데이터로 변환됩니다.
X_train_scaled = (X_train - mean_on_train) / std_on_train
# (훈련 데이터의 평균과 표준 편차를 이용해) 같은 변환을 테스트 세트에도 합니다
X_test_scaled = (X_test - mean_on_train) / std_on_train
 
mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, y_train)

print("훈련 세트 정확도: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("테스트 세트 정확도: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# 행 : 30, 열 : 100, 밝은 색은 큰 양수, 어두운 색은 음수
mlp.coefs_[0].std(axis=1), mlp.coefs_[0].var(axis=1)
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("hidden unit")
plt.ylabel("input feature")
plt.colorbar()
plt.show()

# 신경망
# 장점 : 대규모의 데이터를 다룰 수 있다, 매우 복잡한 모델도 만들 수 있다.\
# 단점 : 종종 학습이 너무 오래 걸림, 데이터 전처리에 주의
# SVM처럼 모든 특성이 같은 의미를 가진 동질의 데이터에서 잘 작동, 다른 종류는 데이터는 트리 모델이 더 나을 수 있음
# 학습된 가중치나 계수의 수가 모델의 복잡도를 계산하는데 도움이 됨
# 특성 100개와 은닉 유닛 100개의 이진 분류 = (입력층, 첫 번째 은닉층)->100*100=10000
# (은닉층, 출력층)->100*1=100
# 따라서, 총 10100개
# 은닉층 하나 더 추가 시, 20100개
# 신경망 매개변수 조정방법 -> 충분히 과대적합 -> 신경망 구조 줄이기, 규제 강화(alpha값 증가)
# 층의 개수, 층당 유닛 개수, 규제, 비선형성 -> 요소들 사용해서 원하는 모델 정의
# solver로 매개변수 학습에 사용되는 알고리즘을 지정할 수 있음
# 기본 값=adam 또는 1bfgs(안정적이지만 규모가 큰 모델이나 대량의 데이터셋에서는 시간이 오래 걸림)
# sgd 옵션등을 이용해 튜닝이 더 가능 