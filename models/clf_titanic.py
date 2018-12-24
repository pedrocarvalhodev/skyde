import os 
import json
import numpy as np
import pandas as pd
import dill as pickle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore")

path = "../data/titanic/"

data_path="https://raw.githubusercontent.com/ahmedbesbes/How-to-score-0.8134-in-Titanic-Kaggle-Challenge/master/data/"

def build_and_train():

    # 1. import data
	#data = pd.read_csv(data_path+'train.csv')
    def get_combined_data():
        # reading train data
        train = pd.read_csv(data_path+'train.csv')
        # reading test data
        test = pd.read_csv(data_path+'test.csv')
        # extracting and then removing the targets from the training data 
        targets = train.Survived
        train.drop(['Survived'], 1, inplace=True)
        # merging train data and test data for future feature engineering
        # we'll also remove the PassengerID since this is not an informative feature
        combined = train.append(test)
        combined.reset_index(inplace=True)
        combined.drop(['index', 'PassengerId'], inplace=True, axis=1)
        return combined

    data = get_combined_data()
	
    # 2. Validate
	X_train, X_test, y_train, y_test = train_test_split(data[pred_var], data['Loan_Status'], test_size=0.25, random_state=42)
	y_train = y_train.replace({'Y':1, 'N':0}).as_matrix()
	y_test = y_test.replace({'Y':1, 'N':0}).as_matrix()

	X_test.to_csv(path+"X_test.csv", index=False)

    # 3. Set pipeline
	pipe = make_pipeline(PreProcessing(),
                         FeatSelection(),
						 RandomForestClassifier())

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

    def status(feature):
        print('Processing', feature, ': ok')

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


    def get_titles():
        # we extract the title from each name
        combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
        # a map of more aggregated title
        # we map each title
        combined['Title'] = combined.Title.map(Title_Dictionary)
        status('Title')
        return combined

    def fill_age(row):
        condition = (
            (grouped_median_train['Sex'] == row['Sex']) & 
            (grouped_median_train['Title'] == row['Title']) & 
            (grouped_median_train['Pclass'] == row['Pclass'])) 
        return grouped_median_train[condition]['Age'].values[0]

    def process_age():
        global combined
        # a function that fills the missing values of the Age variable
        combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
        status('age')
        return combined

    def process_names():
        global combined
        # we clean the Name variable
        combined.drop('Name', axis=1, inplace=True)
        # encoding in dummy variable
        titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
        combined = pd.concat([combined, titles_dummies], axis=1)
        # removing the title variable
        combined.drop('Title', axis=1, inplace=True)
        status('names')
        return combined

    def process_fares():
        global combined
        # there's one missing fare value - replacing it with the mean.
        combined.Fare.fillna(combined.iloc[:891].Fare.mean(), inplace=True)
        status('fare')
        return combined

    def process_embarked():
        global combined
        # two missing embarked values - filling them with the most frequent one in the train  set(S)
        combined.Embarked.fillna('S', inplace=True)
        # dummy encoding 
        embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
        combined = pd.concat([combined, embarked_dummies], axis=1)
        combined.drop('Embarked', axis=1, inplace=True)
        status('embarked')
        return combined

    def process_cabin():
        global combined    
        # replacing missing cabins with U (for Uknown)
        combined.Cabin.fillna('U', inplace=True)
        # mapping each Cabin value with the cabin letter
        combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])
        # dummy encoding ...
        cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')    
        combined = pd.concat([combined, cabin_dummies], axis=1)

        combined.drop('Cabin', axis=1, inplace=True)
        status('cabin')
        return combined

    def process_sex():
        global combined
        # mapping string values to numerical one 
        combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})
        status('Sex')
        return combined

    def process_pclass():
        global combined
        # encoding into 3 categories:
        pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
        # adding dummy variable
        combined = pd.concat([combined, pclass_dummies],axis=1)
        # removing "Pclass"
        combined.drop('Pclass',axis=1,inplace=True)
        status('Pclass')
        return combined

    def process_ticket():
        global combined
        # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
        def cleanTicket(ticket):
            ticket = ticket.replace('.','')
            ticket = ticket.replace('/','')
            ticket = ticket.split()
            ticket = map(lambda t : t.strip(), ticket)
            ticket = filter(lambda t : not t.isdigit(), ticket)
            if len(ticket) > 0:
                return ticket[0]
            else: 
                return 'XXX'

        # Extracting dummy variables from tickets:
        combined['Ticket'] = combined['Ticket'].map(cleanTicket)
        tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
        combined = pd.concat([combined, tickets_dummies], axis=1)
        combined.drop('Ticket', inplace=True, axis=1)
        status('Ticket')
        return combined

    def process_family():
        global combined
        # introducing a new feature : the size of families (including the passenger)
        combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
        
        # introducing other features based on the family size
        combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
        combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
        combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
        
        status('family')
        return combined


    def transform(self, data):
        """Transformations
        """
        combined = get_combined_data()
        combined = get_titles()
        combined = process_names()
        combined = process_age()
        combined = process_fares()
        combined = process_embarked()
        combined = process_cabin()
        combined = process_sex()
        combined = process_pclass()
        combined = process_ticket()
        combined = process_family()
        return combined


class FeatSelection(BaseEstimator, TransformerMixin):
    """Custom Pre-Processing estimator for our use-case
    """
    def __init__(self):
        pass






        ## END
        #return df.as_matrix()

    def fit(self, X, y=None, **fit_params):
        return self

if __name__ == '__main__':
	model = build_and_train()

	filename = 'random_forest_classifier.pk'
	with open(path+filename, 'wb') as file:
		pickle.dump(model, file)