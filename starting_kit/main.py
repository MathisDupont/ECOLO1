model_dir = 'sample_code_submission/'                        # Change the model to a better one once you have one!
result_dir = 'sample_result_submission/' 
problem_dir = 'ingestion_program/'  
score_dir = 'scoring_program/'
from sys import path; path.append(model_dir); path.append(problem_dir); path.append(score_dir); 

import seaborn as sns; sns.set()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



import model
import pickle
import numpy as np
from os.path import isfile
from data_manager import DataManager


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import warnings

with warnings.catch_warnings():
# Uncomment the next lines to auto-reload libraries (this causes some problem with pickles in Python 3)
    import seaborn as sns; sns.set()
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import matplotlib.pyplot as plt
    import pandas as pd
    data_dir = 'public_data'          # The sample_data directory should contain only a very small subset of the data
    data_name = 'plankton'
    from data_io import write
    from model import model
    from libscores import get_metric
    import numpy as np



if __name__=="__main__":
    import matplotlib
    matplotlib.rcParams['backend'] = 'Qt5Agg'
    matplotlib.get_backend()
    D = DataManager(data_name, data_dir)
    model_name = ["Decision Tree", "Nearest Neighbor", "ExtraTreesClassifier", "RandomForestClassifier", "AdaBoost", "QDA"]

    model_list = [
    DecisionTreeClassifier(max_depth=10),
    KNeighborsClassifier(1),
    ExtraTreesClassifier(),
    RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2,random_state=1),
    AdaBoostClassifier(),
    QuadraticDiscriminantAnalysis(),]

    X_train = D.data['X_train']
    Y_train = D.data['Y_train']
    launchModel = model()

    launchModel.fit(X_train, Y_train)
    
    