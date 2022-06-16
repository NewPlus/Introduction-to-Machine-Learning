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

from sklearn.cluster import AgglomerativeClustering # AgglomerativeCluster
from scipy.cluster.hierarchy import dendrogram, ward # hierarchy_cluster(dendrogram, ward)

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

# DBSCAN
# cluster를 미리 지정할 필요가 없다, 복잡한 형상도 찾을 수 있음, 어떤 클래스에도 속하지 않는 포인트 구분가능
# 병합군집 or k-means 보다는 느리지만 비교적 큰 데이터 셋도 가능함
# DBSCAN은 데이터가 많아 붐비는 지역의 포인트 찾음 -> 밀집지역(Dense Region) -> 밀집 지역 포인트를 핵심샘플이라 함
# 매개변수 eps거리 안에 데이터가 매개변수 min_samples 개수 만큼 들어 있으면 이 데이터 포인트를 핵심 샘플로 분류
# if eps안에 포인트 수가 min_samples보다 적다면, 그 포인트는 어떤 클래스에도 속하지 않는 잡음(noise)로 취급
# if eps안에 포인트 수가 min_samples보다 많다면, 그 포인트는 핵심 샘플로 레이블하고 새로운 클러스터 레이블 할당 -> 핵심 샘플이면 그 포인트의 이웃을 차례로 방문
# 클러스터는 eps안에 더 이상 핵심 샘플이 없을 때까지 자라난다.
# 포인트 종류 : 핵심, 경계(eps 거리 안에 있는 핵심 포인트), 잡음 포인트
# 경계 포인트는 같은 데이터 셋이라도 실행 순서(방문 순서)에 따라 달라질 수 있다.

from sklearn.cluster import DBSCAN
X, y = make_blobs(random_state=0, n_samples=12)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
print("클러스터 레이블:\n", clusters)
mglearn.plots.plot_dbscan()
plt.show()

# 색칠한 포인트 : 클러스터에 속한 포인트
# 흰색 포인트 : 잡음 포인트
# eps증가시, 하나의 클러스터에 더 많은 포인트 포함 or 여러 클러스터 병합 등
# mins_samples 증가시, 핵심 포인트 감소, 잡음 포인트 증가

# eps는 가까운 포인트 범위 결정 -> eps가 너무 작으면 어떤 포인트도 핵심포인트 불가함 -> 모든 포인트가 잡음
# eps를 매우 크게 하면 모든 포인트가 단 하나의 클러스터에 속함  

# 덜 조밀한 지역에 있는 포인트들이 mins_samples 수보다 작은 클러스터들은 잡음 포인트가 될지 하나의 클러스터가 될지 결정
# mins_samples는 클러스터의 최소 크기 결정
# eps가 간접적으로 클러스터 개수를 제어 -> StandardScaler나 MinMaxScaler로 모든 특성의 스케일을 비슷한 범위로 조정하는게 좋다

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# 평균이 0, 분산이 1이 되도록 데이터의 스케일을 조정합니다
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)
# 클러스터 할당을 표시합니다
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60, edgecolors='black')
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.show()