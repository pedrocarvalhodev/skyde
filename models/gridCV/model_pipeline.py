import os 
import json
import numpy as np
import pandas as pd
import dill as pickle


from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline

from models.gridCV.preprocess import PreProcessing
from models.gridCV.feat_engineering import FeatEngineering
from models.gridCV.feat_selection import FeatSelection

import warnings
warnings.filterwarnings("ignore")

path = "/home/pedro/repos/ml_web_api/ml-app-model/data/gridCV/"

#data_path="https://raw.githubusercontent.com/ahmedbesbes/How-to-score-0.8134-in-Titanic-Kaggle-Challenge/master/data/"
data_path="/home/pedro/repos/ml_web_api/How-to-score-0.8134-in-Titanic-Kaggle-Challenge/data/"

# 2. Validate
train = pd.read_csv(data_path+'train.csv')
train.drop("PassengerId", axis=1, inplace=True)
y = "Survived"

def ml_pipeline(train, target):

	# 1. Define X, y and split
	X = [x for x in train.columns if x != target]
	X_train, X_test, y_train, y_test = train_test_split(train[X], train[y], test_size=0.5, random_state=42)
	print("Shape: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	y_train = y_train.as_matrix()
	y_test = y_test.as_matrix()

	# 2. Set pipeline
	pipe = make_pipeline(PreProcessing(),
						 FeatEngineering(),
						 FeatSelection(),
						 RandomForestClassifier())

	print("B - Grid shape: ", X_train.shape)

	# 3. Set model parameters
	param_grid = {"randomforestclassifier__n_estimators" : [10, 20, 30],
				 "randomforestclassifier__max_depth" : [None, 6, 8, 10],
				 "randomforestclassifier__max_leaf_nodes": [None, 5, 10, 20], 
				 "randomforestclassifier__min_impurity_split": [0.1, 0.2, 0.3]}

	grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)

	# 4. Fit model
	grid.fit(X_train, y_train)

	return(grid)


if __name__ == '__main__':
	model = ml_pipeline(train=train, target=y) # include option regressor, classifiers or null for features

	filename = 'gridCV.pk'
	with open(path+filename, 'wb') as file:
		pickle.dump(model, file)