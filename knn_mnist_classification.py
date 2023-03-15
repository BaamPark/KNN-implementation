import math
import numpy as np  
from download_mnist import load
import operator  
import time
from collections import Counter

# classify using kNN  
#x_train = np.load('../x_train.npy')
#y_train = np.load('../y_train.npy')
#x_test = np.load('../x_test.npy')
#y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)

def l2_distance(sample1, sample2):
    return np.sqrt(np.sum((sample1 - sample2) ** 2))  # definition of l2

def kNNClassify(newInput, dataSet, labels, k):
    result=[]
    distances = []
    for i in range(newInput.shape[0]):
        row = []
        for j in range(dataSet.shape[0]):
            row.append(l2_distance(newInput[i], dataSet[j]))
        distances.append(row)
    distances = np.array(distances)

    nearest_indices = np.argsort(distances, axis=1)  # sort each row and return index. By sorting in ascending order, first element will be the nearst
    nearest_indices = nearest_indices[:, :k]  # slice it until k
    nearest_labels = labels[nearest_indices]  # take the nearest neighbors' labels. the shape follows nearest_indices
    # return np.array([Counter(neighbors).most_common(1)[0][0] for neighbors in nearest_labels])
    for neighbors in nearest_labels:
        result.append(Counter(neighbors).most_common(1)[0][0])
    result = np.array(result)
    return result

start_time = time.time()
outputlabels=kNNClassify(x_test[0:100],x_train,y_train,5)
result = y_test[0:100] - outputlabels
result = (1 - np.count_nonzero(result)/len(outputlabels))
print(result)
print ("---classification accuracy for knn on mnist: %s ---" %result)
print ("---execution time: %s seconds ---" % (time.time() - start_time))
