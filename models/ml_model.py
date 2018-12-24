import os 
import json
import numpy as np
import pandas as pd
#import dill as pickle

from sklearn.pipeline import Pipeline
import sklearn.datasets as datasets

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore")

path = "/home/pedro/repos/ml_web_api/model-deployment-flask/data/boston_housing/"


def build_and_train():

    df = datasets.load_boston()
    data = pd.DataFrame(df.data, columns=df.feature_names)
    data["medianValue"] = df.target
    y = "medianValue"
    X = [x for x in data.columns if x != y]

    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y], test_size=0.2, random_state=42)
    
    y_train = y_train.as_matrix()
    y_test = y_test.as_matrix()
    #X_train.to_csv(path+"X_train.csv", index=False)
    #X_test.to_csv(path+"X_test.csv", index=False)
    #y_train.to_csv(path+"y_train.csv", index=False)
    #y_test.to_csv(path+"y_test.csv", index=False)

    #pipeline = Pipeline([
    #    ('tfidf',PreProcessing()),
    #    ('clf',LinearRegression())
    #    ])

    #pipeline.fit(X_train,y_train)
    #return(pipeline)

    pipe = make_pipeline(PreProcessing(),
                        RandomForestRegressor())

    #print("-"*60)
    #print(pipe.get_params().keys())
    #print("-"*60)
    #dict_keys(['memory', 'steps', 'preprocessing', 'randomforestregressor', 
    #    'randomforestregressor__bootstrap', 'randomforestregressor__criterion', 
    #    'randomforestregressor__max_depth', 'randomforestregressor__max_features', 
    #    'randomforestregressor__max_leaf_nodes', 'randomforestregressor__min_impurity_decrease', 
    #    'randomforestregressor__min_impurity_split', 'randomforestregressor__min_samples_leaf', 
    #    'randomforestregressor__min_samples_split', 'randomforestregressor__min_weight_fraction_leaf', 
    #    'randomforestregressor__n_estimators', 'randomforestregressor__n_jobs', 'randomforestregressor__oob_score', 
    #    'randomforestregressor__random_state', 'randomforestregressor__verbose', 'randomforestregressor__warm_start'])

    

    ## RandomForestClassifier
    #param_grid = {"randomforestclassifier__n_estimators" : [10, 20, 30],
    #              "randomforestclassifier__max_depth" : [None, 6, 8, 10],
    #              "randomforestclassifier__max_leaf_nodes": [None, 5, 10, 20], 
    #              "randomforestclassifier__min_impurity_split": [0.1, 0.2, 0.3]}
    #
    #grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)

    ## RandomForestRegressor
    param_grid = { 
            "randomforestregressor__n_estimators"      : [10,200,3000],
            "randomforestregressor__max_features"      : ["auto", "sqrt", "log2"],
            "randomforestregressor__min_samples_split" : [2,4,8],
            "randomforestregressor__bootstrap"         : [True, False],
           }


    #grid = GridSearchCV(pipe, param_grid2)

    grid = GridSearchCV(pipe, param_grid) #, n_jobs=-1, cv=5)


    grid.fit(X_train, y_train)

    return(grid)


class PreProcessing(BaseEstimator, TransformerMixin):
    """Custom Pre-Processing estimator for our use-case
    """

    def __init__(self):
        pass

    def transform(self, df):
        """Regular transform() that is a help for training, validation & testing datasets
           (NOTE: The operations performed here are the ones that we did prior to this cell)
        """

        print("Before processing", df.columns)
        df_norm = (df - df.mean()) / (df.max() - df.min())
        df_norm = df_norm.apply(lambda x : np.around(x,1))
        df_norm.columns = [x+"_norm" for x in df.columns]
        df = df.merge(df_norm, how='inner', left_index=True, right_index=True)
        print("After processing", df.columns)
        
        return df.as_matrix()

    # just return self
    def fit(self, X, y=None, **fit_params):
        return self

if __name__ == '__main__':
    clf = build_and_train()
    joblib.dump(clf, 'pipe_model_boston.pkl')