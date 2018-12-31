import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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
		df_norm = df_norm.apply(lambda x : np.around(x,2))
		df_norm.columns = [x+"_norm" for x in df.columns]
		df = df.merge(df_norm, how='inner', left_index=True, right_index=True)
		df_norm=None

		return df

	def fit(self, X, y=None, **fit_params):
		return self