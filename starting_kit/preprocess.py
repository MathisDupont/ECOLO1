from sys import argv
import warnings
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    #from model import model
    from sklearn.base import BaseEstimator
    from data_manager import DataManager
    from sklearn.ensemble import IsolationForest  # Used for outliers detection
    from sklearn.decomposition import PCA  # Used for dimension reduction
    from sklearn.feature_selection import SelectKBest  # Used for feature selection
    from sklearn.feature_selection import chi2  # Used for feature selection
    from libscores import get_metric


class Preprocessor(BaseEstimator):
    def __init__(self):
        self.best_features_nb = 199
        self.best_dim_nb = 10
        self.skbest = SelectKBest(chi2, k=self.best_features_nb)
        self.pca = PCA(n_components=self.best_dim_nb)
        self.features_scores = []
        self.pca_scores = []
        self.detected_outliers = []

    def fit(self, X, y):
        return self.pca.fit(self.skbest.fit(X, y), y)

    def fit_transform(self, X, y=None):
        X1 = self.skbest.fit_transform(X, y)
        return self.pca.fit_transform(X1, y)

    def transform(self, X, y=None):
        return self.pca.transform(self.skbest.transform(X))

    # DO NOT USE THIS FUNCTION UNLESS YOU HAVE A POWERFULL COMPUTER OR/AND A LOT OF TIME OR USE A HIGH SPEED (>4)
    def find_best_params(self, speed):
        print(speed)
        scores = [[0] * 200] * 200
        Y_train = D.data['Y_train']
        for i in range(1, 200, speed):
            #M = model(classifier=DecisionTreeClassifier(max_depth=10, max_features='sqrt', random_state=42))
            feature_selection = SelectKBest(chi2, k=i)
            feature_selection.fit(D.data['X_train'], Y_train)
            X_train = feature_selection.transform(D.data['X_train'])
            for j in range(1, 200, speed):
                tmpM = M
                pca = PCA(n_components=j)
                pca.fit(D.data['X_train'], Y_train)
                X_train = pca.transform(D.data['X_train'])
                tmpM.fit(X_train, Y_train)
                Y_hat_train = tmpM.predict(X_train)
                metric_name, scoring_function = get_metric()
                scores[i][j] = (scoring_function(Y_train, Y_hat_train))
        max_pos = np.argmax(scores)
        self.best_features_nb = max_pos // 200
        self.best_dim_nb = max_pos % 200
        print(self.best_features_nb, self.best_dim_nb)

    def find_best_features(self):
        for i in range(1, 200, 1):
            #M = model(classifier=DecisionTreeClassifier(max_depth=10, max_features='sqrt', random_state=42))
            feature_selection = SelectKBest(chi2, k=i)  # after few tests, 170 look like the best sample
            feature_selection.fit(D.data['X_train'], D.data['Y_train'])
            X_train = feature_selection.transform(D.data['X_train'])
            Y_train = D.data['Y_train']
            M.fit(X_train, Y_train)
            Y_hat_train = M.predict(X_train)
            metric_name, scoring_function = get_metric()
            self.features_scores.append(scoring_function(Y_train, Y_hat_train))
        self.best_features_nb = self.features_scores.index(max(self.features_scores))

    def detect_outliers(self, x):
        rng = np.random.RandomState(42)
        clf = IsolationForest(max_samples=10000, random_state=rng)  # IsolationForest is a choice
        clf.fit(x)
        return clf.predict(x)

    def removeOutliers(self, data):
        outliers = self.detect_outliers(data)
        return data[outliers > 0]

    def find_best_pca(self):
        scores = []
        for i in range(1, 200, 1):
            #M = model(classifier=DecisionTreeClassifier(max_depth=10, max_features='sqrt', random_state=42))
            pca = PCA(n_components=i)
            pca.fit(D.data['X_train'], D.data['Y_train'])
            X_train = pca.transform(D.data['X_train'])
            Y_train = D.data['Y_train']
            M.fit(X_train, Y_train)
            Y_hat_train = M.predict(X_train)
            metric_name, scoring_function = get_metric()
            self.pca_scores.append(scoring_function(Y_train, Y_hat_train))
        self.best_dim_nb = self.pca_scores.index(max(self.pca_scores))

    def apply_parameters(self):
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
        self.find_best_features()
        self.find_best_pca()
        self.detect_outliers()
        self.apply_parameters()

    def plots(self):
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
        output_dir = argv[2];

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
