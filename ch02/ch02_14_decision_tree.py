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

from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer() # 위스콘신 유방암 데이터 셋
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
# X_train(x축 훈련 데이터), X_test(x축 테스트 데이터), y_train(y축 훈련 데이터), y_test(y축 테스트 데이터)
tree = DecisionTreeClassifier(random_state=0)
# 결정트리 분류
tree.fit(X_train, y_train)
# 모델 학습
print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
# 결정트리 분류(사전 가지치기 적용, 연속된 질문을 최대 4개로 제한 -> 과대적합 감소)
tree.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

from sklearn.tree import plot_tree

plot_tree(tree, class_names=["negative", "positive"], feature_names=cancer.feature_names, impurity=False, filled=True, rounded=True, fontsize=4)
plt.show()