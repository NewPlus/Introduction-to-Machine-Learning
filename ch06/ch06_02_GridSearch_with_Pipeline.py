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

pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
pipe.fit(X_train, y_train)
print("테스트 점수: {:.2f}".format(pipe.score(X_test, y_test)))

# 그리드 서치에 파이프라인 적용하기
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
# '단계이름__매개변수이름'의 규칙으로 적용
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print("최상의 교차 검증 정확도: {:.2f}".format(grid.best_score_))
print("테스트 세트 점수: {:.2f}".format(grid.score(X_test, y_test)))
print("최적의 매개변수:", grid.best_params_)

# 이전에 본 그리드 서치와 다른점
# 교차 검증의 각 분할에 MinMaxScaler가 훈련 폴드에 매번 적용되어
# 매개변수 검색 과정에 검증 폴드의 정보가 누설되지 않은 것
# 다음 그것을 묘사한 그림 
mglearn.plots.plot_proper_processing()
plt.show()

# 파이프라인 인터페이스
def fit(self, X, y):
    X_transformed = X
    for name, estimator in self.steps[:-1]:
        # 마지막 단계를 빼고 fit과 transform을 반복합니다
        X_transformed = estimator.fit_transform(X_transformed, y)
    # 마지막 단계 fit을 호출합니다
    self.steps[-1][1].fit(X_transformed, y)
    return self
def predict(self, X):
    X_transformed = X
    for step in self.steps[:-1]:
        # 마지막 단계를 빼고 transform을 반복합니다
        X_transformed = step[1].transform(X_transformed)
    # 마지막 단계 predict을 호출합니다
    return self.steps[-1][1].predict(X_transformed)


# make_pipleline을 사용한 파이프라인 생성
from sklearn.pipeline import make_pipeline
# 표준적인 방법
pipe_long = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC(C=100))])
# 간소화된 방법
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))
print("파이프라인 단계:\n", pipe_short.steps)
# make_pipleline은 굳이 단계명을 붙일 필요가 없거나
# 파이프라인을 표준적인 방법을 쓰기가 번거로울 때 사용한다.
# make_pipleline에서 각 단계는 클래스 이름의 소문자 버전이다. 

pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
print("파이프라인 단계:\n", pipe.steps)
# 만약 같은 이름의 클래스가 두 번 이상 파이프라인에 있는 경우
# 순서대로 클래스소문자이름-1, 클래스소문자이름-2, ... 이런식으로
# 붙여진다. 의미있는 단계 이름을 원한다면 표준적인 방식으로 파이프라인을 구성한다.

# 단계 속성에 접근하기
# cancer 데이터셋에 앞서 만든 파이프라인을 적용합니다
pipe.fit(cancer.data)
# "pca" 단계의 두 개 주성분을 추출합니다
components = pipe.named_steps["pca"].components_
print("components.shape:", components.shape)

# 그리드 서치 안의 파이프라인의 속성에 접근하기
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=4)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
# 가장 최적화된 매개변수는 grid.best_estimator_에 저장되어 있음.
print("최상의 모델:\n", grid.best_estimator_)
print("로지스틱 회귀 단계:\n", grid.best_estimator_.named_steps["logisticregression"])
print("로지스틱 회귀 계수:\n", grid.best_estimator_.named_steps["logisticregression"].coef_)
