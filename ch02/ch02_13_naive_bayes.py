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

# 나이브 베이즈 분류기(Naive Bayes Classifier)
# GaussianNB(연속적인 데이터), BernoulliNB(이진 데이터, 텍스트), MultinomialNB(카운트 데이터, 텍스트)

X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])
counts = {}
for label in np.unique(y):
    # 각 클래스에 대해 반복
    # 특성마다 1 이 나타난 횟수를 센다.
    counts[label] = X[y == label].sum(axis=0)
print("특성 카운트:\n", counts)

# BernoulliNB, MultinomialNB : alpha
# 알고리즘의 모든 특성에 양의 값을 가진 가상의 데이터 포인트를 alpha 개수만큼 추가
# -> alpha가 커지면 통계 데이터가 완만해지고 모델의 복잡도가 낮아짐
# alpha가 성능 향상에 크게 기여 X -> 어느 정도만 높일 수 있다
# GaussianNB는 대부분 고차원인 데이터셋에서 사용, 다른 두 모델은 텍스트 같은 희소한 데이터 카운트에 사용
# 장단점은 선형 모델과 비슷하지만 희소한 고차원 데이터에서 잘 작동
# 비교적 매개변수에 민감 X
# 선형으로 너무 오래 걸리면 나이브 베이즈 모델을 시도할만 함.