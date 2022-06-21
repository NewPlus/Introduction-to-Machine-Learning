import sys
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='NanumGothic') # For Windows 한글 폰트
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

# 교차 검증(Cross Validation) 개념 묘사
mglearn.plots.plot_cross_validation()
plt.show()

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression(max_iter=1000)

scores = cross_val_score(logreg, iris.data, iris.target)
print("5-겹 교차 검증 점수:", scores)

# cv 매개변수는 폴드(fold)의 수를 나타냄
# cv=10이면 10-겹, default는 5-겹(적어도 5-겹 이상을 권함) 
scores = cross_val_score(logreg, iris.data, iris.target, cv=10)
print("10-겹 교차 검증 점수:", scores)
print("교차 검증 평균 점수: {:.2f}".format(scores.mean()))

from sklearn.model_selection import cross_validate
res = cross_validate(logreg, iris.data, iris.target, return_train_score=True)
print(res)
res_df = pd.DataFrame(res)
print(res_df)
print("평균 시간과 점수:\n", res_df.mean())

# 교차 검증의 장점
# 데이터를 무작위로 나눌 때 운 좋게 훈련 세트에는 분류하기 어려운 샘플만 담기게 되었을 경우
# 테스트 세트는 분류하기 쉬운 샘플 뿐이므로 테스트 세트의 정확도는 비현실적으로 높음
# 반대로 운 나쁘게 분류하기 어려운 샘플들이 모두 테스트 세트에 들어간 경우 정확도가 아주 낮음
# 교차 검증 시, 테스트 세트에 각 샘플이 정확하게 한 번씩 들어감 -> 각 샘플은 폴드 중 하나에 속하고 각 폴드는 한 번씩 테스트 세트가 됨
# 교차 검증의 점수를 (그리고 평균값을) 높이기 위해서는 데이터 셋에 있는 모든 샘플에 대해 모델이 잘 일반화 되어야함.
# 분할 한번보다 데이터를 더 효과적으로 사용할 수 있음.(5-겹 교차 검증은 4/5(80%)를, 10-겹 교차 검증은 9/10(90%)를 모델 학습에 사용)
# 주요 단점 : 모델을 k개 만들어야 하므로 데이터를 한 번 나눈 것보다 대략 k배 느림. 

from sklearn.datasets import load_iris
iris = load_iris()
print("Iris 레이블:\n", iris.target)
# 계층별(Stratified) k-겹 교차 검증 및 클래스 레이블 순서대로 정렬한 데이터의 기본 교차 검증 묘사도
mglearn.plots.plot_stratified_cross_validation()
plt.show()

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)
print("교차 검증 점수:\n", cross_val_score(logreg, iris.data, iris.target, cv=kfold))

# 기본 3-겹 교차 검증은 별로 좋지 못함
kfold = KFold(n_splits=3)
print("교차 검증 점수:\n", cross_val_score(logreg, iris.data, iris.target, cv=kfold))
# 각 폴드가 iris 데이터셋의 클래스 중 하나에 대응 -> 아무것도 학습 불가함
# 해결책 : 데이터를 섞어서 샘플 순서를 뒤죽박죽으로 만듦 -> KFold의 shuffle 매개변수를 True로, random_state를 고정하면 똑같은 작업 재현(이렇게 안하면 폴드가 매번 바뀜) 

# LOOCV(Leave-One-Out cross-validation)
# 폴드 1개에 샘플 1개만 들어있는 k-겹 교차 검증
# 각 반복에서 하나의 데이터 포인트를 선택해 테스트 세트로 사용함
# 큰 데이터셋에서는 시간이 많이 걸리지만, 작은 데이터 셋에서는 이따금 좋은 결과 보여줌 
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("교차 검증 분할 횟수: ", len(scores))
print("평균 정확도: {:.2f}".format(scores.mean()))

# 임의 분할 교차 검증 묘사도
mglearn.plots.plot_shuffle_split()
plt.show()

# 임의 분할 교차 검증
# train_size 만큼의 포인트를 훈련 세트로 만들고
# test_size 만큼의 (훈련 세트와 중첩되지 않는) 포인트로 테스트 세트를 만들도록 분할함
# 이 분할은 n_splits 횟수만큼 반복됨
# 반복 횟수를 훈련 세트나 테스트 세트의 크기와 독립적으로 조절해야 할 경우 유용함
# 또한 test_size와 train_size의 합을 전체와 다르게 함으로써 전체 데이터 일부만 사용할 수 있음.
# 이런 데이터 부분 샘플링은 대규모 데이터 셋으로 작업할 때 도움이 됨.  
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("교차 검증 점수:\n", scores)

# 그룹별 교차 검증
# 데이터 안에 매우 연관된 그룹이 있을 때도 교차 검증을 널리 사용함
# 데이터 셋에 없는 경우에도 모델이 정확하게 구분할 수 있는 분류기를 만드는 것이 목표임. 
from sklearn.model_selection import GroupKFold
# 인위적 데이터셋 생성
X, y = make_blobs(n_samples=12, random_state=0)
# 처음 세 개의 샘플은 같은 그룹에 속하고
# 다음은 네 개의 샘플이 같습니다.
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups=groups, cv=GroupKFold(n_splits=3))
print("교차 검증 점수:\n", scores)
mglearn.plots.plot_group_kfold()
plt.show()
