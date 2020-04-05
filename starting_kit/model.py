"""
Created on Sat Apr 4 
Last revised: Apr 5, 2020

@author: Antoine Barbannaud, Minh Kha Nguyen

This class is based on zClassifier.py by Isabelle Guyon.
This class contains the classifier with the integration of a preprocesor
"""

import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from libscores import get_metric
from data_manager import DataManager
import matplotlib
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

import preprocess as prepro

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

model_dir = 'sample_code_submission/'                        # Change the model to a better one once you have one!
result_dir = 'sample_result_submission/' 
problem_dir = 'ingestion_program/'  
score_dir = 'scoring_program/'
data_dir = 'public_data'          
data_name = 'plankton'



class model(BaseEstimator):
    '''Class based on MonsterClassifier'''
    def __init__(self):
        '''
        fancy_classifier = Pipeline([
                    ('preprocessing', Preprocessor()),
                    ('classification', RandomForestClassifier(n_estimators=136, max_depth=None, min_samples_split=2, random_state=0))
                    ])
        self.clf = VotingClassifier(estimators=[
                    ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
                    ('Gaussian Classifier', GaussianNB()),
                    ('Support Vector Machine', SVC(probability=True)),
                    ('Fancy Classifier', fancy_classifier)],
                    voting='soft')   
        '''
        self.mdl = RandomForestClassifier(n_estimators=136, max_depth=None, min_samples_split=2, random_state=0)
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.prep = prepro.Preprocessor()
        
    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        #1, y1 = self.prep.fit_transform(X)
        X1 = self.prep.fit_transform(X, y)
        return self.mdl.fit(X1, y)

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        X1 = self.prep.transform(X)
        return self.mdl.predict(X1)

    def predict_proba(self, X):
        ''' Similar to predict, but probabilities of belonging to a class are output.'''
        return self.mdl.predict_proba(X) # The classes are in the order of the labels returned by get_classes
        
    def save(self, path="./"):
        file = open(path + '_model.pickle', "wb")
        pickle.dump(self, file)
        file.close()

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        file.close()
        return self
 
   
def test():
    '''
    Trains the model and returns its score
    '''
    matplotlib.rcParams['backend'] = 'Qt5Agg'
    matplotlib.get_backend()
    D = DataManager(data_name, data_dir)
    #Load le model
    mdl = model()
    
    Prepro = prepro.Preprocessor()
    #D.data['X_train'] = Prepro.removeOutliers(D.data['X_train'])
    #D.data['Y_train'] = Prepro.removeOutliers(D.data['Y_train'])
    X_train = D.data['X_train']
    Y_train = D.data['Y_train'].ravel()

    #test de l'entrainement
    mdl.fit(X_train, Y_train)

    #test de la prediction
    Y_hat_train = mdl.predict(D.data['X_train']) 
    Y_hat_valid = mdl.predict(D.data['X_valid'])
    Y_hat_test = mdl.predict(D.data['X_test'])

    metric_name, scoring_function = get_metric()
    scores = cross_val_score(mdl, X_train, Y_train, cv=5, scoring=make_scorer(scoring_function))
    print('\nCV score (95 perc. CI): %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
        
        
    
if __name__=="__main__":
    test()