import numpy as np
from collections import Counter

class KNN:
    def __init__(self, X_train, y_train, k=3):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    @staticmethod
    def l2_distance(sample1, sample2):
        return np.sqrt(np.sum((sample1 - sample2) ** 2)) #definition of l2

    def predict(self, X_test):
        dists = []
        for i in range(X_test.shape[0]):
            row = []
            for X_train in self.X_train:
                row.append(self.l2_distance(X_test[i], X_train))
            dists.append(row)
        dists = np.array(dists) #shape: (test[i], train[i]) n^2

        nearest_indices = np.argsort(dists, axis=1) #sort each row and return index. By sorting in ascending order, first element will be the nearst
        nearest_indices = nearest_indices[:, :self.k] #slice it until k
        nearest_labels = self.y_train[nearest_indices] #take the nearest neighbors' labels. the shape follows nearest_indices
        return np.array([Counter(neighbors).most_common(1)[0][0] for neighbors in nearest_labels]) #return top most votes array


if __name__ == '__main__':
    X_train = np.array([[0, 1, 0], [0, 1, 1], [1, 2, 1], [1, 2, 0],
                        [1, 2, 2], [2, 2, 2], [1, 2, -1], [2, 2, 3],
                        [-1, -1, -1], [0, -1, -2], [0, -1, 1], [-1, -2, 1]])

    y_train = np.array([0, 0, 0, 0, #0 denotes class A
                  1, 1, 1, 1, #1 denotes class B
                  2, 2, 2, 2]) #2 denotes class C

    X_test = np.array([[1, 0, 1], [1, 1, 1]])
    X_test = np.array([[1, 0, 1]])

    my_dict = {0: 'A', 1: 'B', 2: 'C'}

    knn = KNN(X_train, y_train, 1)
    prediction1 = knn.predict(X_test)
    print("The [1, 0, 1] is classified to {}, where k is 1".format(my_dict[prediction1[0]]))
    knn = KNN(X_train, y_train, 2)
    prediction2 = knn.predict(X_test)
    print("The [1, 0, 1] is classified to {}, where k is 2".format(my_dict[prediction2[0]]))
    knn = KNN(X_train, y_train, 3)
    prediction3 = knn.predict(X_test)
    print("The [1, 0, 1] is classified to {}, where k is 3".format(my_dict[prediction3[0]]))