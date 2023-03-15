# KNN Classifier Implementation

The K-Nearest Neighbors (KNN) algorithm is a non-parametric, lazy learning method used for classification and regression. The algorithm works by finding the k-nearest neighbors to a given data point, based on a distance metric, and making a prediction based on the majority class or average value of those neighbors.

In this project, I implement the KNN algorithm using only the NumPy library in Python, without relying on the scikit-learn library. By doing so, we gain a deeper understanding of the algorithm's inner workings and can customize it to our specific needs.

## Implementation Details
Our implementation includes the following steps:

1. Loading and preprocessing the data
2. Calculating the distance between the data points
3. Selecting the k-nearest neighbors
4. Making predictions based on the majority class or average value of the neighbors
5. Evaluating the model's performance

## main Files

- `knn_mnist_classification.py`: Implement KNN to classify mnist image
- `knn_3d_data_classification.py`: Implment KNN to classify 3d data point
- `knn_2d_classification.py`: Implement KNN to classify 2d data point

## Usage

To use this script, simply run `mnist_downloader.py`:

```bash
python mnist_downloader.py
