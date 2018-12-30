from sklearn.base import BaseEstimator, TransformerMixin

class CustomPreProcessing(BaseEstimator, TransformerMixin):
	"""Custom Pre-Processing estimator for our use-case
	"""

	def __init__(self):
		pass

	def transform(self, df):

		def process_age(df):
			df['Age'] = df['Age'].fillna(df.groupby(['Sex','Pclass'])['Age'].transform('mean'))
			return df

		def process_names(df):
			df.drop('Name', axis=1, inplace=True)
			titles_dummies = pd.get_dummies(df['Title'], prefix='Title')
			df = pd.concat([df, titles_dummies], axis=1)
			df.drop('Title', axis=1, inplace=True)
			return df


		def process_fares(df):
			df.Fare.fillna(df.Fare.median(), inplace=True)
			return df

		def process_embarked(df):
			df.Embarked.fillna('S', inplace=True)
			embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
			df = pd.concat([df, embarked_dummies], axis=1)
			df.drop('Embarked', axis=1, inplace=True)
			return df

		def process_cabin(df):
			df.Cabin.fillna('U', inplace=True)
			df['Cabin'] = df['Cabin'].map(lambda c: c[0])
			cabin_dummies = pd.get_dummies(df['Cabin'], prefix='Cabin')    
			df = pd.concat([df, cabin_dummies], axis=1)
			df.drop('Cabin', axis=1, inplace=True)
			return df

		def process_sex(df):
			df['Sex'] = df['Sex'].map({'male':1.0, 'female':0.0})
			df['Sex'] = df['Sex'].fillna(1.0)
			return df

		def process_pclass(df):
			pclass_dummies = pd.get_dummies(df['Pclass'], prefix="Pclass")
			df = pd.concat([df, pclass_dummies],axis=1)
			df.drop('Pclass',axis=1,inplace=True)
			return df

		def process_family(df):
			df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
			df['Singleton'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)
			df['SmallFamily'] = df['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
			df['LargeFamily'] = df['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
			df.drop("FamilySize", axis=1, inplace=True)
			return df

		def inner_transform(df):
			df = process_fares(df)
			df = process_age(df)
			df = process_sex(df)
			df = process_pclass(df)
			df = process_family(df)
			return df

		df = inner_transform(df)
		#df = df[["Sex","Age","Pclass","SibSp","Parch","Fare",'Singleton','SmallFamily','LargeFamily']].copy()
		df = df.select_dtypes(include=[np.number]).copy()
		df = df.astype(float)
		return df

	def fit(self, X, y=None, **fit_params):
		return self