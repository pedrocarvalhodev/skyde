import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MatrixTransform(BaseEstimator, TransformerMixin):
	"""Custom Pre-Processing estimator for our use-case
	"""
	def __init__(self):
		pass

	def transform(self, df):
		return df.as_matrix()

	def fit(self, X, y=None, **fit_params):
		return self