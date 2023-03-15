# KNN Classifier Implementation

This repository contains a Python script to download, save, and load the MNIST dataset, a popular dataset of handwritten digits used for training and evaluating machine learning models.
The K-Nearest Neighbors (KNN) algorithm is a non-parametric, lazy learning method used for classification and regression. The algorithm works by finding the k-nearest neighbors to a given data point, based on a distance metric, and making a prediction based on the majority class or average value of those neighbors.

In this project, I implement the KNN algorithm using only the NumPy library in Python, without relying on the scikit-learn library. By doing so, I gain a deeper understanding of the algorithm's inner workings and can customize it to my specific needs.

My implementation includes the following steps:

Loading and preprocessing the data
Calculating the distance between the data points
Selecting the k-nearest neighbors
Making predictions based on the majority class or average value of the neighbors
Evaluating the model's performance
This implementation is suitable for small to medium-sized datasets and serves as a starting point for more advanced implementations.

## main Files

- `knn_mnist_classification.py`: Implement KNN to classify mnist image
- `knn_3d_data_classification.py`: Implment KNN to classify 3d data point
- `knn_2d_classification.py`: Implement KNN to classify 2d data point

## Usage

To use this script, simply run `mnist_downloader.py`:

```bash
python mnist_downloader.py
