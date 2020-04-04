from sys import argv
import warnings
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from model import model
    from sklearn.base import BaseEstimator
    from data_manager import DataManager
    from sklearn.ensemble import IsolationForest  # Used for outliers detection
    from sklearn.decomposition import PCA  # Used for dimension reduction
    from sklearn.feature_selection import SelectKBest  # Used for feature selection
    from sklearn.feature_selection import chi2  # Used for feature selection
    from libscores import get_metric
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import cross_val_score


class Preprocessor(BaseEstimator):
    """Contain all methods of preprocessing :
        - fit and transform the data with default parameters
        - find the best params
        - search the best features number
        - search the best dimensions number
        - detect and remove outliers
        - apply the parameters to the data
    """

    def __init__(self):
        self.best_features_nb = 199  # Best number of features found with find_best_features
        self.best_dim_nb = 10  # Best number of dimension found with find_best_pca
        self.skbest = SelectKBest(chi2, k=self.best_features_nb)
        self.pca = PCA(n_components=self.best_dim_nb)
        self.features_scores = []
        self.pca_scores = []
        self.detected_outliers = []

    def fit(self, X, y):
        """
        Run score function on (X, y) and get the appropriate features and dimension.
        :param X: The input samples.
        :param y: The target values (class labels).
        :return: A new version of this instance with fitted pca et skbest
        """
        self.skbest = self.skbest.fit(X, y)
        self.pca = self.pca.fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.
        :param X: The imput samples
        :param y: The target values (class labels).
        :return: Transformed array.
        """
        return self.pca.fit_transform(self.skbest.fit_transform(X, y), y)

    def transform(self, X, y=None):
        """
        Reduce X to the selected features and dimensions.
        :param X: The imput samples
        :param y: The target values (class labels).
        :return: The input samples with only the selected features and dimensions.
        """
        return self.pca.transform(self.skbest.transform(X))

    def find_best_params(self, speed):
        """
        Search the best dimensions and features number.
        DO NOT USE THIS FUNCTION UNLESS YOU HAVE A POWERFULL COMPUTER OR/AND A LOT OF TIME OR USE A HIGH SPEED (>4)
        :param speed: The number of features and dimension jumped on each loop
        :return: The best dimensions and features number
        """
        print(speed)
        scores = [[0] * 200] * 200
        Y_train = D.data['Y_train'].ravel()
        for i in range(1, 200, speed):
            # M = model selected by the modeling team
            M = model(classifier=RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2, random_state=1))
            feature_selection = SelectKBest(chi2, k=i)
            feature_selection.fit(D.data['X_train'], Y_train)
            X_train = feature_selection.transform(D.data['X_train'])
            for j in range(1, 200, speed):
                tmpM = M
                pca = PCA(n_components=j)
                pca.fit(D.data['X_train'], Y_train)
                X_train = pca.transform(D.data['X_train'])
                tmpM.fit(X_train, Y_train)
                metric_name, scoring_function = get_metric()
                scrs = cross_val_score(M, X_train, Y_train, cv=5, scoring=make_scorer(scoring_function))
                scores[i][j] = (scrs.mean())
        max_pos = np.argmax(scores)
        self.best_features_nb = max_pos // 200
        self.best_dim_nb = max_pos % 200
        print(self.best_features_nb, self.best_dim_nb)

    def find_best_features(self):
        """
        Execute the model with different quantity of features (1 to 200) and return the quantity of features who give the best model's score.
        :return: The best features number
        """
        for i in range(1, 200, 1):
            M = model(classifier=RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2, random_state=1))
            feature_selection = SelectKBest(chi2, k=i)  # after few tests, 170 look like the best sample
            feature_selection.fit(D.data['X_train'], D.data['Y_train'])
            X_train = feature_selection.transform(D.data['X_train'])
            Y_train = D.data['Y_train'].ravel()
            M.fit(X_train, Y_train)
            metric_name, scoring_function = get_metric()
            scores = cross_val_score(M, X_train, Y_train, cv=5, scoring=make_scorer(scoring_function))
            self.features_scores.append(scores.mean())
        self.best_features_nb = self.features_scores.index(max(self.features_scores))

    def detect_outliers(self):
        """
        Detect the outliers from the data and mark down who they are.
        :return: A list containing 1 if the element is not an outlier, -1 else.
        """
        rng = np.random.RandomState(42)
        clf = IsolationForest(max_samples=10000, random_state=rng)  # We choose IsolationForest after lookings on the documentations of differents type of outliers_detectors methodes.
        clf.fit(D.data['X_train'])  # use IsolationForest on the datas
        self.detected_outliers = clf.predict(D.data['X_train']) # mark down the outilers of the datas (detected_outliers= -1 if outlier, 1 else )

    def removeOutliers(self, y, data):
        """
        Remove the ouliers who are detected
        :param y: The list of detected outliers
        :param data: The data of which the outliers will be removed
        :return: The data without outliers
        """
        return data[y > 0]

    def find_best_pca(self):
        """
        Find the best dimensions number using the PCA (Principal Component Analysis).
        :return: The best dimensions number
        """
        scores = []
        for i in range(1, 200, 1):
            M = model(classifier=RandomForestClassifier(n_estimators=116, max_depth=None, min_samples_split=2, random_state=1))
            pca = PCA(n_components=i)
            pca.fit(D.data['X_train'], D.data['Y_train'])
            X_train = pca.transform(D.data['X_train'])
            Y_train = D.data['Y_train'].ravel()
            M.fit(X_train, Y_train)
            metric_name, scoring_function = get_metric()
            scores = cross_val_score(M, X_train, Y_train, cv=5, scoring=make_scorer(scoring_function))
            self.pca_scores.append(scores.mean())
        self.best_dim_nb = self.pca_scores.index(max(self.pca_scores))

    def apply_parameters(self):
        """
        Apply to all data what have been fund as best features and pca and detect and remove the outliers
        :return: Preprocessed data
        """
        feature_selection = SelectKBest(chi2, k=self.best_features_nb).fit(D.data['X_train'], D.data['Y_train'])
        # apply this feature_selection to the data
        D.data['X_train'] = feature_selection.transform(D.data['X_train'])
        D.data['X_valid'] = feature_selection.transform(D.data['X_valid'])
        D.data['X_test'] = feature_selection.transform(D.data['X_test'])
        D.data['X_train'] = self.removeOutliers(self.detected_outliers, D.data['X_train'])
        D.data['Y_train'] = self.removeOutliers(self.detected_outliers, D.data['Y_train'])
        pca = PCA(n_components=self.best_dim_nb).fit(D.data['X_train'], D.data['Y_train'])
        D.data['X_train'] = pca.transform(D.data['X_train'])
        D.data['X_valid'] = pca.transform(D.data['X_valid'])
        D.data['X_test'] = pca.transform(D.data['X_test'])

    def preprocess(self):
        """
        Call all the methods in one
        :return: Preprocessed data
        """
        self.find_best_features()
        self.find_best_pca()
        self.detect_outliers()
        self.apply_parameters()

    def plots(self):
        """
        Visualize the results of the preprocessing
        :return: 3 plots
        """
        plt.plot(self.features_scores)
        plt.plot(self.pca_scores)
        sns.scatterplot(x="sum_axis_1_50", y="variance", data=data, hue=y_pred_train)
        plt.plot()


if __name__ == "__main__":
    # We can use this to run this file as a script and test the Preprocessor
    if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data"
        output_dir = "../results"
    else:
        input_dir = argv[1]
        output_dir = argv[2]

    basename = 'Iris'
    D = DataManager(basename, input_dir)  # Load data
    print("*** Original data ***")
    print(D)

    Prepro = Preprocessor()

    # Preprocess on the data and load it back into D
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])
    D.feat_name = np.array(['PC1', 'PC2'])
    D.feat_type = np.array(['Numeric', 'Numeric'])

    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print(D)
