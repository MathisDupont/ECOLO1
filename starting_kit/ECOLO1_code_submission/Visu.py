'''
Ici vous trouverez les fonctions de visualisation, test en cours
'''

import pickle
import numpy as np   # We recommend to use numpy arrays
import matplotlib.pyplot as plt
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from data_manager import DataManager


class Visu:
    
    def __init__(self):
        self.bonjour =3


    def kmeans(self, X_train, Y_train):
        #on prend nos donnees X_train
        data = scale(X_train)

        n_samples, n_features = data.shape 
        n_digits = len(np.unique(Y_train)) #et nos valeurs dans Y_train

        labels = Y_train[:,0]

        sample_size = 300

        print("n_digits: %d, \t n_samples %d, \t n_features %d"
              % (n_digits, n_samples, n_features))

        # #############################################################################
        # Visualize the results on PCA-reduced data

        reduced_data = PCA(n_components=2).fit_transform(data)
        kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
        kmeans.fit(reduced_data)

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

        R0 = np.zeros((1536,2))
        R1 = np.zeros((1536,2))
        R2 = np.zeros((1536,2))
        R3 = np.zeros((1536,2))
        R4 = np.zeros((1536,2))
        R5 = np.zeros((1536,2))
        R6 = np.zeros((1536,2))

        cpt0=0
        cpt1=0
        cpt2=0
        cpt3=0
        cpt4=0
        cpt5=0
        cpt6=0

        for i in range (len(Y_train)):
            if Y_train[i]==0:
                R0[cpt0][0]=reduced_data[i][0]
                R0[cpt0][1]=reduced_data[i][1]
                cpt0+=1
            elif Y_train[i]==1:
                R1[cpt1][0]=reduced_data[i][0]
                R1[cpt1][1]=reduced_data[i][1]
                cpt1+=1
            elif Y_train[i]==2:
                R2[cpt2][0]=reduced_data[i][0]
                R2[cpt2][1]=reduced_data[i][1]
                cpt2+=1
            elif Y_train[i]==3:
                R3[cpt3][0]=reduced_data[i][0]
                R3[cpt3][1]=reduced_data[i][1]
                cpt3+=1
            elif Y_train[i]==4:
                R4[cpt4][0]=reduced_data[i][0]
                R4[cpt4][1]=reduced_data[i][1]
                cpt4+=1
            elif Y_train[i]==5:
                R5[cpt5][0]=reduced_data[i][0]
                R5[cpt5][1]=reduced_data[i][1]
                cpt5+=1
            elif Y_train[i]==6:
                R6[cpt6][0]=reduced_data[i][0]
                R6[cpt6][1]=reduced_data[i][1]
                cpt6+=1
            
            
        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        fig, ax = plt.subplots(1, figsize=(10,5))


        ax.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        ax.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        #ax.plot(R0[:, 0], R0[:, 1], 'w+', markersize=4)
        #ax.plot(R1[:, 0], R1[:, 1], 'r.', markersize=2)
        #ax.plot(R2[:, 0], R2[:, 1], 'b.', markersize=2)
        #ax.plot(R3[:, 0], R3[:, 1], 'g.', markersize=2)
        #ax.plot(R4[:, 0], R4[:, 1], 'k.', markersize=2)
        #ax.plot(R5[:, 0], R5[:, 1], 'k.', markersize=2)
        #ax.plot(R6[:, 0], R6[:, 1], 'k.', markersize=2)



        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        ax.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
        ax.set_title('K-means \n'
                  'Centroids are marked with white cross')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlabel("First component PC1")
        ax.set_ylabel("Second component PC2")

        
        