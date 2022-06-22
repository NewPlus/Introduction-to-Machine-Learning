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
from sklearn.linear_model import LogisticRegression # LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # 위스콘신 유방암 데이터 셋

from sklearn.datasets import load_boston
boston = load_boston() # Boston 주택 가격 데이터 셋

from sklearn.decomposition import PCA # PCA
from sklearn.decomposition import NMF # NMF
from sklearn.datasets import make_blobs #make_blobs용
from sklearn.cluster import KMeans # k-means
from sklearn.datasets import make_moons #make_moons용
from sklearn.svm import SVC # SVC

from sklearn.cluster import AgglomerativeClustering # AgglomerativeCluster
from scipy.cluster.hierarchy import dendrogram, ward # hierarchy_cluster(dendrogram, ward)
from sklearn.cluster import DBSCAN # DBSCAN
from sklearn.metrics.cluster import silhouette_score # 실루엣 계수
from sklearn.metrics.cluster import adjusted_rand_score # ARI adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier # 랜덤 포레스트

from sklearn.preprocessing import OneHotEncoder # OneHotEncoder
from sklearn.compose import make_column_transformer # make_column_transformer를 사용하여 ColumnTransformer 생성
from sklearn.preprocessing import KBinsDiscretizer # 한 번에 여러 개의 특성에 적용할 수 있고, 기본적으로 구간에 원-핫-인코딩을 적용
from sklearn.preprocessing import PolynomialFeatures # 다항식 추가용
from sklearn.feature_selection import SelectPercentile, f_classif # 단변량 통계
from sklearn.feature_selection import SelectFromModel # 모델 기반 특성 선택
from sklearn.feature_selection import RFE # 반복적 특성 선택
from sklearn.linear_model import Ridge # Ridge

from sklearn.model_selection import cross_val_score # 교차 검증용
from sklearn.model_selection import cross_validate # 교차 검증용
from sklearn.model_selection import LeaveOneOut # LOOCV(Leave-One-Out cross-validation)
from sklearn.model_selection import GridSearchCV # GridSearchCV

from sklearn.datasets import fetch_lfw_people # people 사용 예제용
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

people.target[0:10], people.target_names[people.target[0:10]]

# np.bool은 1.20버전부터 deprecated됩니다. 대신 bool을 사용하세요.
mask = np.zeros(people.target.shape, dtype=bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
    
X_people = people.data[mask]
y_people = people.target[mask]

# 0~255 사이의 흑백 이미지의 픽셀 값을 0~1 사이로 스케일 조정합니다.
# (옮긴이) MinMaxScaler를 적용하는 것과 거의 동일합니다.
X_people = X_people / 255.

# 데이터 적재와 분할
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

# 훈련 데이터의 최솟값, 최댓값을 계산합니다
scaler = MinMaxScaler().fit(X_train)

# 훈련 데이터의 스케일을 조정합니다
X_train_scaled = scaler.transform(X_train)

svm = SVC()
# 스케일 조정된 훈련데이터에 SVM을 학습시킵니다
svm.fit(X_train_scaled, y_train)
# 테스트 데이터의 스케일을 조정하고 점수를 계산합니다
X_test_scaled = scaler.transform(X_test)
print("테스트 점수: {:.2f}".format(svm.score(X_test_scaled, y_test)))

# 데이터 전처리와 매개변수 선택
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print("최상의 교차 검증 정확도: {:.2f}".format(grid.best_score_))
print("테스트 점수: {:.2f}".format(grid.score(X_test_scaled, y_test)))
print("최적의 매개변수: ", grid.best_params_)

# 스케일을 조정한 데이터를 사용하여 SVC의 매개변수에 대해 그리드 서치 수행
# 그러나 데이터의 최솟값과 최댓값을 계산할 때 학습을 위해 훈련 세트에 있는 모든 데이터를 사용함
# 이후 스케일이 조정된 훈련 데이터에서 교차 검증을 사용해 그리드 서치를 수행
# 교차 검증의 각 분할에서 원본 훈련 세트 데이터의 어떤 부분은 훈련 폴드가 되고 어떤 부분은 검증 폴드가 됨
# 검증 폴드는 훈련 폴드로 학습된 모델이 새로운 데이터에 적용될 때의 성능을 측정하는 데 사용
# 그러나 데이터 스케일을 조정할 때 검증 폴드에 들어있는 정보까지 이미 사용함.
# 따라서 새로운 데이터가 들어오면 이 데이터는 훈련 데이터의 스케일 조정에 없었으므로
# 그 최소와 최대가 훈련 데이터와 다를 수 있다. 

mglearn.plots.plot_improper_processing()
plt.show()

# 이것을 해결하기 위해서는 교차 검증의 분할이 모든 전처리 과정보다 앞서야 함.
# 데이터셋의 정보를 이용하는 모든 처리 과정은 데이터셋의 훈련 부분에만 적용되어야하므로 교차 검증안에 있어야함
# 여기에 pipeline 이용 -> 여러 처리 단계를 하나의 scikit-learn 추정기 형태로 묶어주는 파이썬 클래스이다.
# fit, predict, score 메소드 제공(다른 scikit-learn 모델들과 유사하게 동작)
# 가장 일반적인 사용처는 분류기와 같은 지도 학습 모델과 스케일 조정 등 전치리 단계를 연결할 경우임.

# 파이프라인 구축하기
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
pipe.fit(X_train, y_train)
print("테스트 점수: {:.2f}".format(pipe.score(X_test, y_test)))