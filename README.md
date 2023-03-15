# DeepLearningIntroHW1

Deep Learning Intro HW1
This repository contains a Python script to download, save, and load the MNIST dataset, a popular dataset of handwritten digits used for training and evaluating machine learning models.

Files
mnist_downloader.py: The main Python script that downloads, saves, and loads the MNIST dataset.
Usage
To use this script, simply run mnist_downloader.py:

bash
Copy code
python mnist_downloader.py
This will download the MNIST dataset files from Yann LeCun's website and save them as a pickle file (mnist.pkl). The script also contains a load() function that can be used to load the dataset into memory.

python
Copy code
from mnist_downloader import load

training_images, training_labels, test_images, test_labels = load()
Dependencies
Python 3.x
numpy
urllib
