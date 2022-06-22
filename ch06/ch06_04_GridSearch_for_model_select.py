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

from sklearn.pipeline import Pipeline # 파이프라인 구축용
from sklearn.pipeline import make_pipeline # make_pipeline
from sklearn.preprocessing import PolynomialFeatures # 다항식 특성 선택

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

# 모델 선택을 위한 그리드 서치
pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

param_grid = [
    {'classifier': [SVC()], 'preprocessing': [StandardScaler()],
     'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'classifier': [RandomForestClassifier(n_estimators=100)],
     'preprocessing': [None], 'classifier__max_features': [1, 2, 3]}]
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("최적의 매개변수:\n{}\n".format(grid.best_params_))
print("최상의 교차 검증 점수: {:.2f}".format(grid.best_score_))
print("테스트 세트 점수: {:.2f}".format(grid.score(X_test, y_test)))

# 중복 계산 피하기
# 대규모로 그리드 서치를 하다보면 가끔 동일한 단계를 여러 번 수행할 수도 있음.
# 각 설정에 대해 StandardScaler가 다시 만들어짐 -> 비용이 많이 드는 변환을 사용한다면 계산 낭비가 심해짐
# 간단한 해결책으로 계산 결과를 캐싱하기

pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())], memory="cache_folder")

# 단점 2가지
# 1. 실제 디스크에서 읽고 쓰기 위해 직렬화(serializtion)이 필요
# 2. n_jobs 매개변수가 캐싱을 방해 -> 최악의 경우로 캐시되기 전에 n_jobs만큼의 작업 프로세스가 동시에 동일한 계산을 중복으로 수행할 수도 있음
# dask_ml 라이브러리에서 제공하는 GridSearchCV를 사용하면 이런 단점을 모두 피할 수 있음 