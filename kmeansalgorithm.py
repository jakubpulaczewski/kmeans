import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

DATASET_FILE = 'data.csv'
FIRST_COLUMN, SECOND_COLUMN = 'V1','V2'
K_MAX = 7
'''
Computes the euclidean distace between two points (x1,y1) and (x2,y2).

@x1: first point of coordinates (x1,y1)
@x2: the second point of coordinates (x2,y2)
@return: the euclidean distance
'''
def distance(x1,x2,ax = 1):
    return np.linalg.norm(x1-x2,axis=ax)

'''
Randomly initialises the values for centroids.

@X: the matrix of inputs
@k: the number of instances to generate
@return: an array with randomly generated centroids
'''
def get_random_centeroids(X,k):
    x = np.random.randint(0, np.max(X)-np.max(X)/2, size=k) 
    y = np.random.randint(0, np.max(X)-np.max(X)/2, size=k)
    return np.array(list(zip(x1, y)))

def show_plot(C,X):
    colors = ['r', 'g', 'b', 'y', 'c', 'r']
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters_labels[j] == i])
        plt.scatter(points[:, 0], points[:, 1], s=20, c=colors[i], alpha=0.25)
        plt.scatter(C[i,0], C[i,1], marker='P', s=200, c='black')
    plt.draw()
    plt.pause(2)
    plt.clf()

# Step 1: Read Dataset.
data = pd.read_csv(DATASET_FILE)
x1 = data[FIRST_COLUMN].values
x2 = data[SECOND_COLUMN].values
X = list(zip(x1, x2)) 

for k in range(1, K_MAX):
    print("The value for K is {0}".format(k))
    # Step 2: Set K instances from the dataset randomly
    C = get_random_centeroids(X,k=k)
    # To store the value of centroids when it updates
    C_old = np.zeros(C.shape)
    # To store the labels of centroids
    clusters_labels = np.zeros(len(X))
    congervence = distance(C, C_old, None)

    #Until convergence repeat steps 3 and 4
    #convergence = no instances have moved among clusters 
    while congervence != 0:    
        # Step 3: Assign points to closest cluster
        for i in range(len(X)):
            distances_x_centroids = distance(X[i], C)  
            clusters_labels[i] = np.argmin(distances_x_centroids)
        # Storing the old centroid values
        C_old = np.copy(C)
        # Step 4: Compute new mean for each cluster
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters_labels[j] == i]
            C[i] = np.mean(points, axis=0)
        congervence = distance(C, C_old, None)
    show_plot(C,X)
plt.show()    

#Include RuntimeWarning: mean of emtpy slice.
