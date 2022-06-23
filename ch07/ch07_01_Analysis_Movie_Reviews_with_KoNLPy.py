import konlpy
import pandas as pd
import numpy as np

df_train = pd.read_csv('C:/Users/user/Desktop/NLL/Introduction to Machine Learning/ch07/data/ratings_train.txt', delimiter='\t', keep_default_na=False)
print(df_train.head())

# 테스트 데이터 읽어오기
text_train, y_train = df_train['document'].values, df_train['label'].values

df_test = pd.read_csv('C:/Users/user/Desktop/NLL/Introduction to Machine Learning/ch07/data/ratings_test.txt', delimiter='\t', keep_default_na=False)
text_test = df_test['document'].values
y_test = df_test['label'].values

from konlpy.tag import Okt

class PicklableOkt(Okt):

    def __init__(self, *args):
        self.args = args
        Okt.__init__(self, *args)
    
    def __setstate__(self, state):
        self.__init__(*state['args'])
        
    def __getstate__(self):
        return {'args': self.args}

okt = PicklableOkt()

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


param_grid = {'tfidfvectorizer__min_df': [3, 5 ,7],
              'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
              'logisticregression__C': [0.1, 1, 10]}


pipe = make_pipeline(TfidfVectorizer(tokenizer=okt.morphs), LogisticRegression())
grid = GridSearchCV(pipe, param_grid, n_jobs=-1)

# 그리드 서치를 수행합니다
grid.fit(text_train, y_train)
print("최상의 크로스 밸리데이션 점수: {:.3f}".format(grid.best_score_))
print("최적의 크로스 밸리데이션 파라미터: ", grid.best_params_)

X_test = grid.best_estimator_.named_steps["tfidfvectorizer"].transform(text_test)
score = grid.best_estimator_.named_steps["logisticregression"].score(X_test, y_test)
print("테스트 세트 점수: {:.3f}".format(score))
