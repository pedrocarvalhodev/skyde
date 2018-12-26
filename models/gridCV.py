import os 
import json
import numpy as np
import pandas as pd
import dill as pickle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore")

path = "../data/gridCV/"

#data_path="https://raw.githubusercontent.com/ahmedbesbes/How-to-score-0.8134-in-Titanic-Kaggle-Challenge/master/data/"
data_path="/home/pedro/repos/ml_web_api/How-to-score-0.8134-in-Titanic-Kaggle-Challenge/data/"

def build_and_train():

	# 2. Validate
	train = pd.read_csv(data_path+'train.csv')
	train.drop("PassengerId", axis=1, inplace=True)

	y = "Survived"
	X = [x for x in train.columns if x != y]

	X_train, X_test, y_train, y_test = train_test_split(train[X], train[y], test_size=0.5, random_state=42)

	print("Shape: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	y_train = y_train.as_matrix()
	y_test = y_test.as_matrix()

	# 3. Set pipeline
	pipe = make_pipeline(PreProcessing(),
						 FeatEngineering(),
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

		Title_Dictionary = {
				"Capt": "Officer",
				"Col": "Officer",
				"Major": "Officer",
				"Jonkheer": "Royalty",
				"Don": "Royalty",
				"Sir" : "Royalty",
				"Dr": "Officer",
				"Rev": "Officer",
				"the Countess":"Royalty",
				"Mme": "Mrs",
				"Mlle": "Miss",
				"Ms": "Mrs",
				"Mr" : "Mr",
				"Mrs" : "Mrs",
				"Miss" : "Miss",
				"Master" : "Master",
				"Lady" : "Royalty"}

		def status(feature):
			#print('Processing', feature, ': ok')
			pass


		def get_titles(df):
			# we extract the title from each name
			df['Title'] = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
			# a map of more aggregated title
			# we map each title
			df['Title'] = df.Title.map(Title_Dictionary)
			status('Title')
			return df

		def process_age(df):

			df['Age'] = df['Age'].fillna(df.groupby(['Sex','Pclass'])['Age'].transform('mean'))
			#status('age')
			return df

		def process_names(df):
			#global df
			# we clean the Name variable
			df.drop('Name', axis=1, inplace=True)
			# encoding in dummy variable
			titles_dummies = pd.get_dummies(df['Title'], prefix='Title')
			df = pd.concat([df, titles_dummies], axis=1)
			# removing the title variable
			df.drop('Title', axis=1, inplace=True)
			status('names')
			return df

		def process_fares(df):
			#global df
			# there's one missing fare value - replacing it with the mean.
			df.Fare.fillna(df.Fare.median(), inplace=True)
			status('fare')
			return df

		def process_embarked(df):
			df.Embarked.fillna('S', inplace=True)
			# dummy encoding 
			embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
			df = pd.concat([df, embarked_dummies], axis=1)
			df.drop('Embarked', axis=1, inplace=True)
			status('embarked')
			return df

		def process_cabin(df):
			df.Cabin.fillna('U', inplace=True)
			# mapping each Cabin value with the cabin letter
			df['Cabin'] = df['Cabin'].map(lambda c: c[0])
			# dummy encoding ...
			cabin_dummies = pd.get_dummies(df['Cabin'], prefix='Cabin')    
			df = pd.concat([df, cabin_dummies], axis=1)

			df.drop('Cabin', axis=1, inplace=True)
			status('cabin')
			return df

		def process_sex(df):
			#global df
			# mapping string values to numerical one 
			df['Sex'] = df['Sex'].map({'male':1.0, 'female':0.0})
			df['Sex'] = df['Sex'].fillna(1.0)
			status('Sex')
			return df

		def process_pclass(df):
			#global df
			# encoding into 3 categories:
			pclass_dummies = pd.get_dummies(df['Pclass'], prefix="Pclass")
			# adding dummy variable
			df = pd.concat([df, pclass_dummies],axis=1)
			# removing "Pclass"
			df.drop('Pclass',axis=1,inplace=True)
			status('Pclass')
			return df


		def process_family(df):
			#global df
			# introducing a new feature : the size of families (including the passenger)
			df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
			
			# introducing other features based on the family size
			df['Singleton'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)
			df['SmallFamily'] = df['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
			df['LargeFamily'] = df['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
			df.drop("FamilySize", axis=1, inplace=True)
			
			status('family')
			return df


		def transform(df):
			"""Transformations
			"""
			##df = get_titles(df)
			##df = process_names(df)
			df = process_age(df)
			df = process_fares(df)
			##df = process_embarked(df)
			##df = process_cabin(df)
			
			df = process_sex(df)
			
			##df = process_pclass(df)
			##df = process_ticket(df)
			#df = process_family(df)
			return df

		df = transform(df)
		df = df[["Sex","Age","Pclass","SibSp","Parch","Fare"]].copy()
		df = df.select_dtypes(include=[np.number]).copy()
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
		df_norm = df_norm.apply(lambda x : np.around(x,1))
		df_norm.columns = [x+"_norm" for x in df.columns]
		df = df.merge(df_norm, how='inner', left_index=True, right_index=True)
		df_norm=None
		df.replace([np.inf, -np.inf], np.nan, inplace=True)

		main_vars = ["Sex", "Age", "Pclass","SibSp","Parch","Fare","Pclass_norm","SibSp_norm","Parch_norm","Fare_norm"]
		df = df[main_vars].copy()
		return df.as_matrix()


	def fit(self, X, y=None, **fit_params):
		return self


if __name__ == '__main__':
	model = build_and_train()

	filename = 'gridCV.pk'
	with open(path+filename, 'wb') as file:
		pickle.dump(model, file)