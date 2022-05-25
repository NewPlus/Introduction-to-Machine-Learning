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

from sklearn.datasets import load_boston
boston = load_boston()
print("데이터의 형태:", boston.data.shape)
# 506 data-points, 13 features
# 13개의 입력특성만을 고려한 dataset

X, y = mglearn.datasets.load_extended_boston()
print("X.shape:", X.shape)
# 506 data-points, 104 features
# feature engineering(특성 공학)
# 13(원래 특성) + 91(13개에서 2개씩 중복포함한 곱) = 104개의 특성
# 13개에서 2개씩 중복포함한 곱이란?
# 13(처음 13개의 교차항) + 12(첫 번째 특성을 제외한 12개 교차항) + 11 + ... + 1 = 91
# 또는 ((n, k)) = (n+k-1, k) 따라서, ((13, 2)) = (13+2-1, 2) = 14!/(2!*(14-2)!) = 91
# 104개의 특성을 고려한 dataset