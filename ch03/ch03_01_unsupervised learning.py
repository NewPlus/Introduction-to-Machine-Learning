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

# 인위적으로 만든 이진 분류 데이터셋
# 1번 특성(x축 : 10~15), 2번 특성(y축 : 1~9)
plt.rcParams['image.cmap'] = "gray"
mglearn.plots.plot_scaling()
# scikit-learn의 StandardScaler, RobustScaler, MinMaxScaler, Normalizer
# StandardScaler : 각 특성의 평균을 0, 분산을 1로 변경하여 모든 특성이 같은 크기를 갖도록 함.(단, 특성의 최소, 최대 크기를 제한하지는 않음)
# RobustScaler : 특성들이 같은 스케일을 갖게 되는 점(StandardScaler 유사), 평균과 분산 대신 중간 값(median)과 사분위 값(quartile)을 사용, 전체 데이터와 아주 동떨어진 데이터 포인트(ex. 특정 에러, 이상치outlier -> 다른 스케일 조정에서는 문제일 수 도)에 영향 안 받음. 
# MinMaxScaler : 모든 특성이 정확하게 0과 1 사이에 위치하도록 데이터를 변경, ex. 2차원이면 모든 데이터가 x 축의 0과 1, y 축의 0과 1 사이의 사각 영역에 담김.
# Normalizer : 좀 특이함, 특성 벡터의 유클리디안 길이(거리)가 1이 되도록 데이터 포인트 조정(지름이 1인 원(2차원) -> 3차원이면 구) -> 각 데이터 포인트가 다른 비율로(길이에 반비례) 스케일 조정 -> 데이터의 방향만 중요할 때 주로 사용함.
# <공식들>
# StandardScaler : (x-(x평균)) / 표준편차 -> 표준점수 혹은 z-점수
# RobustScaler : (x-q2) / (q3-q1) -> q2는 중간값, q1은 1사분위 값(x보다 작은 수가 전체 개수의 1/4인 x), q3는 3사분위 값(x보다 큰 수가 전체 개수의 1/4인 x)
# MinMaxScaler : (x-x_min) / (x_max-x_min) -> 데이터에서 최솟값을 빼고 전체 범위로 나눔.
# Normalizer : norm 매개변수(l1, l2(유클리디안 거리), max) -> 행(데이터 포인트)마다 각각 정규화.
plt.show()

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

# 데이터 변환
X_train_scaled = scaler.transform(X_train)
# 스케일이 조정된 후 데이터셋의 속성을 출력합니다
print("변환된 후 크기:", X_train_scaled.shape)
print("스케일 조정 전 특성별 최소값:\n", X_train.min(axis=0))
print("스케일 조정 전 특성별 최대값:\n", X_train.max(axis=0))
print("스케일 조정 후 특성별 최소값:\n", X_train_scaled.min(axis=0))
print("스케일 조정 후 특성별 최대값:\n", X_train_scaled.max(axis=0))

# 테스트 데이터 변환
X_test_scaled = scaler.transform(X_test)
# 스케일이 조정된 후 테스트 데이터의 속성을 출력합니다
print("스케일 조정 후 특성별 최소값:\n", X_test_scaled.min(axis=0))
print("스케일 조정 후 특성별 최대값:\n", X_test_scaled.max(axis=0))
