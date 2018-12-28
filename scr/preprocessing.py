import pandas as pd

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