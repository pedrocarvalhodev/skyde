import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatSelection(BaseEstimator, TransformerMixin):
	"""Custom Pre-Processing estimator for our use-case
	"""
	def __init__(self):
		pass

	def transform(self, df):
		main_vars = ["Sex", "Age","Fare", "Age_norm","Fare_norm"
					 'Singleton','SmallFamily','LargeFamily',
					 'Pclass_1','Pclass_2','Pclass_3']

		for x in main_vars:
			if x not in df.columns:
				df[x] = 0.0

		df = df[main_vars].copy()
		return df

	def fit(self, X, y=None, **fit_params):
		return self