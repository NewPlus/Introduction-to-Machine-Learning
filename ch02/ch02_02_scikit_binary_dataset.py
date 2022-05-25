import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import scipy as sp
import IPython
import sklearn
import mglearn

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # 위스콘신 유방암 데이터 셋
print("cancer.keys():\n", cancer.keys())
# 569 data-points, 30 features
print("유방암 데이터의 형태:", cancer.data.shape)
# 종양은 양성(benign, 해롭지 않은 종양)과 음성(malignant, 암 종양)으로 구분
# 즉, 이진 분류 데이터 셋
# 212 malignant, 357 benign
print("클래스별 샘플 갯수:\n", {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})
print("특성 이름:\n", cancer.feature_names)