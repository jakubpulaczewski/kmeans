# Libraries
import matplotlib.pyplot as plt                        
import numpy as np                                      
import math                                             
from scipy.special import comb                

# Constants
FEATURE_SIZE = 300                            # The size of the features
NUMBER_CLASSES = 4                            # The number of classses (animals, fruits, veggies and countries)

# Distance algorithms
EUCLIDEAN = "euclidean"                       # Euclidean Distance
MANHATAN = "manhattan"                        # Manhattan Distance
COSINE = "cosine"                             # Cosine similarity rule


class Cluster():
    ''' 
    @ centroids: the number of clusters
    @ size_array: an array that contains the size of each class
    @ dataset:  the dataset containing all the classes.
    '''
    def __init__(self, centroids, size_array, dataset,):
        self.centroids = centroids 
        self.size_array = size_array
        self.dataset = dataset

    # Return the number of combination of A things taken at b. 
    # Also often expressed as "A choose b"
    def myComb(self, a,b):
        return comb(a, b,exact=True)

    # Computes the precision, recall and F-score for each set of clusters.
    def evaluate(self, matrix):
        vComb = np.vectorize(self.myComb)                                           # Returns a vectorized function
        positives = vComb(matrix.sum(axis=0),2).sum()                               # Returns a sum of all positives (true positives and false positives)
        negatives = vComb(matrix.sum(axis=1),2).sum()                               # Returns a sum of all negatives (true negatives and false negatives)
        true_positives = vComb(matrix, 2).sum()                                     
        false_positives = positives - true_positives
        false_negatives = negatives - true_positives
        # Clustering Evaluation  
        precision = float((true_positives) / (true_positives + false_positives))    
        recall = float((true_positives) / (true_positives + false_negatives))
        f_score = float((2.0 * precision * recall) / (precision + recall))
        return precision,recall, f_score

    # Take a random unique point from the dataset and set it as a cluster
    # This was done to make sure that there is no empty clusters as it can occur with random values.
    def initialization(self):
        return self.dataset[np.random.choice(self.dataset.shape[0], self.centroids, replace=False), :]

    # Return a euclidean distance between two points (matrices) a and b.
    def euclideanDistance(self, a1, a2, ax = 1):
        return np.sqrt(((a1-a2)**2).sum(axis = ax))      

    # Return a manhattan distance between two points (matrices) a and b
    def manhattanDistance(self, a, b, ax = 1):
        return (abs(a-b).sum(axis = ax))

    # Normalizes the arrays (used for cosine similarity)
    def norm(self, array, array2,ax):
        a = (array**2).sum(axis = ax)
        b = (array2**2).sum(axis = ax)
        return np.sqrt(a*b)

    # Returns a consine similarity distance between two points a and b
    def cosineSimilarity(self, a, b, ax = 1):
        norm_ab = self.norm(a,b,ax)
        dot = (a*b).sum(axis = ax)
        return 1 - (dot / (norm_ab))


    #Assigns points to each cluster and then evaluates the clustering
    def assignment(self,distance):
        cluster_values = np.zeros((NUMBER_CLASSES,self.dataset.shape[0]))             # Creates a numpy array of shape (4,329) to store all the clusters values
        centroids_array = np.array(self.initialization())                             # Initializes random points for the clusters             
        old_centroids_array = np.zeros(centroids_array.shape)                         # Initializes old cluster array
        cluster_labels = np.zeros(self.dataset.shape[0])                              # This array will store all the labels for clusters
        
        # Distance algorithms (EUCLIDEAN, MANHATTAN and COSINE)
        if distance == EUCLIDEAN:
            errorDistance = self.euclideanDistance(centroids_array,old_centroids_array, None)
        elif distance == MANHATAN:
            errorDistance = self.manhattanDistance(centroids_array,old_centroids_array, None)
        elif distance == COSINE:
            errorDistance = self.cosineSimilarity(centroids_array,old_centroids_array, None)

        #Convererge if two arrays (old cluster array and new one) are equal (and therefore equal to 0)
        while errorDistance != 0: 

            for i in range(self.dataset.shape[0]):
                # Resizes from (300,) to (1,300)
                data = np.reshape(self.dataset[i],(1,300))                                                  
                # Calculate which cluster is the closest to the each data point in the dataset.
                if distance == EUCLIDEAN:                                                                                                                                       
                    distances =  self.euclideanDistance(data,centroids_array)
                elif distance == MANHATAN:
                    distances =  self.manhattanDistance(data,centroids_array)
                elif distance == COSINE:
                    distances =  self.cosineSimilarity(data,centroids_array)
                 # so that it will not be 0 but 1 to indicate the class 1, 2 to indicate the class 2 etc.
                cluster_labels[i] = np.argmin(distances) + 1 
            # The new array becomes the old ones.
            old_centroids_array = np.copy(centroids_array)                                                   

            # Calculates the new position by computing the mean of all the points.
            for i in range(self.centroids):
                points = [self.dataset[j] for j in range(self.dataset.shape[0]) if cluster_labels[j] == i+1]
                centroids_array[i] = np.mean(points,axis = 0)

            # Computing the error distance. If it's 0, it has converged.
            if distance == EUCLIDEAN:
                errorDistance = self.euclideanDistance(centroids_array,old_centroids_array, None)
            elif distance == MANHATAN:
                errorDistance = self.manhattanDistance(centroids_array,old_centroids_array, None)
            elif distance == COSINE:
                errorDistance = self.cosineSimilarity(centroids_array,old_centroids_array, None)
        # Assigns the labels to each class (animals, fruits, veggies and countries)
        next_size = prev_size = 0
        for index in range(NUMBER_CLASSES):
            prev_size = next_size
            next_size = next_size + self.size_array[index]
            cluster_values[index:index + 1, 0 : self.size_array[index]] = cluster_labels[prev_size:next_size]

        # Calculates occurences of classes in each cluster and reshapes it to (4,X) where X is a number of clusters
        dots =[np.count_nonzero(cluster_values[i] == j+1) for i in range(NUMBER_CLASSES) for j in range(self.centroids)]
        cooccurrence_matrix = np.reshape(dots,(NUMBER_CLASSES,self.centroids)) 
        #Evaluates and returns precision, recall and f_score                          
        precision, recall, f_score = self.evaluate(cooccurrence_matrix)
        return precision, recall, f_score

#Read the file and then splits the data and creates a numpy array
#The first column (names of animals etc.) is deleted so that the array can be float
#Returns the data of a given file
def readData(file):
    with open(file) as myfile:
        data = myfile.read()                                                                      
        data = data.splitlines()                                                                  
        data = [x.split(' ') for x in data]                                                      
        data = np.array(data)                                                                     
        data = np.delete(data, 0, axis=1)                                                         
        data = np.array(data, dtype = float)                                                     
        return data

def graph(precision, recall, f_score):
    plt.plot(precision,label="precision")
    plt.plot(recall,label="recall")
    plt.plot(f_score,label="f_score")
    plt.xticks(np.arange(10), np.arange(1, 11))
    plt.xlabel("K (clusters) : ")
    plt.ylabel("Score: ")
    plt.legend()
    plt.show()

#Euclidean distance with  - question2
def question2(size_list, dataset):
    precision = np.zeros(10)                             
    recall = np.zeros(10)                                       
    f_score = np.zeros(10)       
    for i in range(1,11):
        #print("K: ", i)
        cluster = Cluster(i,size_list, dataset)
        precision[i-1],recall[i-1],f_score[i-1] = cluster.assignment(EUCLIDEAN)
    graph(precision, recall, f_score)

#Euclidean distance with normalised feature vector. - Question3
def question3(size_list, normalized_dataset):
    precision = np.zeros(10)                             
    recall = np.zeros(10)                                       
    f_score = np.zeros(10)  
    for i in range(1,11):
        #print("K: ", i)
        cluster = Cluster(i,size_list, normalized_dataset)
        precision[i-1],recall[i-1],f_score[i-1] = cluster.assignment(EUCLIDEAN)
    graph(precision, recall, f_score)

#Manhattarn distance over unnormalised feature vectors - Question 4
def question4(size_list, dataset):
    precision = np.zeros(10)                             
    recall = np.zeros(10)                                       
    f_score = np.zeros(10)  

    for i in range(1,11):
        #print("K: ", i)
        cluster = Cluster(i,size_list, dataset)
        precision[i-1],recall[i-1],f_score[i-1] = cluster.assignment(MANHATAN)
    graph(precision, recall, f_score)

#Manhattan distance with normalised feature vectors. q5
def question5(size_list, normalized_dataset):
    precision = np.zeros(10)                             
    recall = np.zeros(10)                                       
    f_score = np.zeros(10)  

    for i in range(1,11):
        #print("K: ", i)
        cluster = Cluster(i,size_list, normalized_dataset)
        precision[i-1],recall[i-1],f_score[i-1] = cluster.assignment(MANHATAN)
    graph(precision, recall, f_score)

#Cosine similarity over unnommralised feature vectors q6
def question6(size_list, dataset):
    precision = np.zeros(10)                             
    recall = np.zeros(10)                                       
    f_score = np.zeros(10)  

    for i in range(1,11):
        #print("K: ", i)
        cluster = Cluster(i,size_list, dataset)
        precision[i-1],recall[i-1],f_score[i-1] = cluster.assignment(COSINE)
    graph(precision, recall, f_score)

if __name__ == '__main__':   
    # Reading the data files.
    animals = readData("animals")
    fruits = readData("fruits")
    countries = readData("countries")
    veggies = readData("veggies")

    # Combining the data files into one dataset.
    dataset = np.concatenate((animals, fruits, veggies, countries), axis=0)

    # Size of each list (used for labeling the classes to each cluster)
    size_List =np.array([animals.shape[0], fruits.shape[0], veggies.shape[0], countries.shape[0]])

    # Normalized dataset
    nomalized_data = [(dataset[i][j]) / math.sqrt(sum(dataset[i]**2)) for i in range (dataset.shape[0]) for j in range (FEATURE_SIZE)]
    normalized_dataset = np.reshape(nomalized_data,(dataset.shape[0],FEATURE_SIZE))

    ''' Assignment Question '''
    #Question 2
    print("Question 2")
    question2(size_List, dataset)

    #Question 3
    print("Question 3")
    question3(size_List, normalized_dataset)

    #Question 4
    print("Question 4")
    question4(size_List, dataset)

    # Question 5
    print("Question 5")
    question5(size_List, normalized_dataset)

    # Question 6
    print("Question 6")
    question6(size_List, dataset)
