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
from models.gridCV.matrix_transform import MatrixTransform

import warnings
warnings.filterwarnings("ignore")

path = "/home/pedro/repos/ml_web_api/ml-app-model/data/gridCV/"

#data_path="https://raw.githubusercontent.com/ahmedbesbes/How-to-score-0.8134-in-Titanic-Kaggle-Challenge/master/data/"
data_path="/home/pedro/repos/ml_web_api/How-to-score-0.8134-in-Titanic-Kaggle-Challenge/data/"

# 2. Validate
train = pd.read_csv(data_path+'train.csv')
train.drop("PassengerId", axis=1, inplace=True)
y = "Survived"

def ml_pipeline(train, target, ml_type):

	# 1. Define X, y and split
	X = [x for x in train.columns if x != target]
	X_train, X_test, y_train, y_test = train_test_split(train[X], train[y], test_size=0.5, random_state=42)
	print("Shape: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	y_train = y_train.as_matrix()
	y_test = y_test.as_matrix()



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
		grid.fit(X_train, y_train)
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
		grid.fit(X_train, y_train)
		return(grid)

	elif ml_type == "Features":
		print("preprocess")
		PreProcessingInst = PreProcessing()
		X_train = PreProcessingInst.transform(df=X_train)
		print("feat_eng")
		FeatEngineeringInst = FeatEngineering()
		X_train = FeatEngineeringInst.transform(df=X_train)
		print("feat_selection")
		FeatSelectionInst = FeatSelection()
		X_train = FeatSelectionInst.transform(df=X_train)
		#X_train.to_csv(path+"features_data.csv")
		print("Downloaded features data.")
		return X_train
	
	else:
		print("Warning: ml_type error")
		return None


#if __name__ == '__main__':
#	model = ml_pipeline(train=train, target=y, ml_type=ml_type) # include option regressor, classifiers or null for features
#
#	filename = 'gridCV.pk'
#	with open(path+filename, 'wb') as file:
#		pickle.dump(model, file)