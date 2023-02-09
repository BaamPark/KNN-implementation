import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter


# load mini training data and labels
mini_train = np.load('knn_minitrain.npy') #(40, 2)
mini_train_label = np.load('knn_minitrain_label.npy')

# randomly generate test data
mini_test = np.random.randint(20, size=20)
mini_test = mini_test.reshape(10,2)


def l2_distance(sample1, sample2):
    return np.sqrt(np.sum((sample1 - sample2) ** 2))  # definition of l2

# Define knn classifier
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
#[2 3 2 2 1 1 3 0 3 1]

outputlabels=kNNClassify(mini_test,mini_train,mini_train_label,5)

print(mini_train.shape)
print(mini_train_label.shape)

print ('random test points are:', mini_test)
print ('knn classfied labels for test:', outputlabels)

# plot train data and classfied test data
train_x = mini_train[:,0]
train_y = mini_train[:,1]
fig = plt.figure()
plt.scatter(train_x[np.where(mini_train_label==0)], train_y[np.where(mini_train_label==0)], color='red')
plt.scatter(train_x[np.where(mini_train_label==1)], train_y[np.where(mini_train_label==1)], color='blue')
plt.scatter(train_x[np.where(mini_train_label==2)], train_y[np.where(mini_train_label==2)], color='yellow')
plt.scatter(train_x[np.where(mini_train_label==3)], train_y[np.where(mini_train_label==3)], color='black')

test_x = mini_test[:,0]
test_y = mini_test[:,1]
outputlabels = np.array(outputlabels)
plt.scatter(test_x[np.where(outputlabels==0)], test_y[np.where(outputlabels==0)], marker='^', color='red')
plt.scatter(test_x[np.where(outputlabels==1)], test_y[np.where(outputlabels==1)], marker='^', color='blue')
plt.scatter(test_x[np.where(outputlabels==2)], test_y[np.where(outputlabels==2)], marker='^', color='yellow')
plt.scatter(test_x[np.where(outputlabels==3)], test_y[np.where(outputlabels==3)], marker='^', color='black')

# save diagram as png file
plt.savefig("miniknn.png")
