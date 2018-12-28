import os 
import json
import numpy as np
import pandas as pd
import dill as pickle

import warnings

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

try:
	from preprocess import process_age, process_fares, process_sex, process_family
except ImportError:
	from .preprocess import process_age, process_fares, process_sex, process_family

warnings.filterwarnings("ignore")

path = "/home/pedro/repos/ml_web_api/ml-app-model/data/gridCV/"

#data_path="https://raw.githubusercontent.com/ahmedbesbes/How-to-score-0.8134-in-Titanic-Kaggle-Challenge/master/data/"
data_path="/home/pedro/repos/ml_web_api/How-to-score-0.8134-in-Titanic-Kaggle-Challenge/data/"

# 2. Validate
train = pd.read_csv(data_path+'train.csv')
train.drop("PassengerId", axis=1, inplace=True)
y = "Survived"

def ml_pipeline(train, target):

	X = [x for x in train.columns if x != target]

	X_train, X_test, y_train, y_test = train_test_split(train[X], train[y], test_size=0.5, random_state=42)

	print("Shape: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	y_train = y_train.as_matrix()
	y_test = y_test.as_matrix()

	# 3. Set pipeline
	pipe = make_pipeline(PreProcessing(),
						 FeatEngineering(),
						 FeatSelection(),
						 RandomForestClassifier())

	print("B - Grid shape: ", X_train.shape)
	param_grid = {"randomforestclassifier__n_estimators" : [10, 20, 30],
				 "randomforestclassifier__max_depth" : [None, 6, 8, 10],
				 "randomforestclassifier__max_leaf_nodes": [None, 5, 10, 20], 
				 "randomforestclassifier__min_impurity_split": [0.1, 0.2, 0.3]}

	grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)

	# 4. Fit model
	grid.fit(X_train, y_train)

	return(grid)


class PreProcessing(BaseEstimator, TransformerMixin):
	"""Custom Pre-Processing estimator for our use-case
	"""

	def __init__(self):
		pass

	def transform(self, df):
		def inner_transform(df):
			df = process_fares(df)
			df = process_age(df)
			df = process_sex(df)
			df = process_family(df)
			return df

		df = inner_transform(df)
		df = df[["Sex","Age","Pclass","SibSp","Parch","Fare",'Singleton','SmallFamily','LargeFamily']].copy()
		#df = df.select_dtypes(include=[np.number]).copy()
		df = df.astype(float)
		return df

	def fit(self, X, y=None, **fit_params):
		return self


class FeatEngineering(BaseEstimator, TransformerMixin):
	"""Custom Pre-Processing estimator for our use-case
	"""
	def __init__(self):
		pass

	def transform(self, df):
		"""Features selection
		"""
		df = df.reset_index(drop=True)
		df_norm = (df - df.mean()) / (df.max() - df.min())
		#df_norm = df_norm.apply(lambda x : np.around(x,1))
		df_norm.columns = [x+"_norm" for x in df.columns]
		df = df.merge(df_norm, how='inner', left_index=True, right_index=True)
		df_norm=None
		#df.replace([np.inf, -np.inf], np.nan, inplace=True)
		return df



	def fit(self, X, y=None, **fit_params):
		return self

class FeatSelection(BaseEstimator, TransformerMixin):
	"""Custom Pre-Processing estimator for our use-case
	"""
	def __init__(self):
		pass

	def transform(self, df):
		main_vars = ["Sex", "Age", "Pclass","SibSp","Parch","Fare",'Singleton','SmallFamily','LargeFamily',
					 "Pclass_norm","SibSp_norm","Parch_norm","Fare_norm"]
		df = df[main_vars].copy()
		return df.as_matrix()

	def fit(self, X, y=None, **fit_params):
		return self

if __name__ == '__main__':
	model = ml_pipeline(train=train, target=y)

	filename = 'gridCV.pk'
	with open(path+filename, 'wb') as file:
		pickle.dump(model, file)