import os 
#import json
import numpy as np
import pandas as pd
import dill as pickle

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from models.gridCV.preprocess import PreProcessing
from models.gridCV.feat_engineering import FeatEngineering
from models.gridCV.feat_selection import FeatSelection
from models.gridCV.matrix_transform import MatrixTransform

import warnings
warnings.filterwarnings("ignore")

path = "/home/pedro/repos/ml_web_api/ml-app-model/data/gridCV/"


def ml_pipeline(train, target, ml_type):

	# 1. Define X, y and split
	X = [x for x in train.columns if x != target]
	X_train, X_test, y_train, y_test = train_test_split(train[X], train[target], test_size=0.5, random_state=42)
	print("Shape: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	y_train = y_train.reset_index(drop=False)

	for col in X_train.columns:
		if col in ["PassengerId", "index", "Unnamed: 0"]:
			X_train.drop(col, axis=1, inplace=True)

	# 2. Set pipeline

	if ml_type == "Classifier":
		pipe = make_pipeline(PreProcessing(),
							 FeatEngineering(),
							 FeatSelection(),
							 MatrixTransform(),
							 RandomForestClassifier())

		param_grid = {"randomforestclassifier__n_estimators" : [10, 20, 30],
					 "randomforestclassifier__max_depth" : [None, 6, 8, 10],
					 "randomforestclassifier__max_leaf_nodes": [None, 5, 10, 20], 
					 "randomforestclassifier__min_impurity_split": [0.1, 0.2, 0.3]}

		grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)
		grid.fit(X_train, y_train[target].as_matrix())
		return(grid)

	elif ml_type == "Regressor":
		print("Regressor method")
		pipe = make_pipeline(PreProcessing(),
							 FeatEngineering(),
							 FeatSelection(),
							 MatrixTransform(),
							 RandomForestRegressor())

		param_grid = {"randomforestregressor__n_estimators": [10, 50],
					  "randomforestregressor__max_depth": [None, 5],
					  "randomforestregressor__max_features": [5, 10],
					  "randomforestregressor__min_samples_split": [5, 10],
					  "randomforestregressor__min_samples_leaf": [3, 10],
					  "randomforestregressor__bootstrap": [True, False]}

		grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)
		grid.fit(X_train, y_train[target].as_matrix())
		return(grid)

	elif ml_type == "LogisticRegression":
		pipe = make_pipeline(PreProcessing(),
							 FeatEngineering(),
							 FeatSelection(),
							 MatrixTransform(),
							 LogisticRegression())

		print("SHAPE: ",X_train.shape, y_train[target].shape)
		pipe.fit(X_train, y_train[target].as_matrix())
		return(pipe)


	elif ml_type == "Features":
		PreProcessingInst = PreProcessing()
		FeatEngineeringInst = FeatEngineering()
		FeatSelectionInst = FeatSelection()

		X_train = PreProcessingInst.transform(df   = X_train)
		X_train = FeatEngineeringInst.transform(df = X_train)
		X_train = FeatSelectionInst.transform(df   = X_train)

		print("Downloaded features data.")
		y_train = y_train[target].to_frame()
		df_feat = X_train.merge(y_train, left_index=True, right_index=True, how="inner")
		return X_train, y_train, df_feat
	
	else:
		print("Warning: ml_type error")
		return None