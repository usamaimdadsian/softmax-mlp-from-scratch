# Softmax Regression & MLP from Scratch

## Table of Contents

- [About](#about)
- [Dataset](#dataset)
- [Unit Tests](#unit_tests)
- [Author](#author)
- [Support](#support)


## About <a name = "about"></a>

This GitHub repository contains a project where I have implemented the algorithms for softmax regression and multi-layer perceptron  MLP using only the numpy library. The goal of this project was to gain a deeper understanding of these algorithms by building them  from scratch, without using any pre-existing machine learning libraries. The repository includes the code for both algorithms, as well as a set of Jupyter notebooks that demonstrate how to use them on MNIST dataset. I hope that this project will be helpful for anyone looking to gain a better understanding of these algorithms and how to implement them using numpy.

## Dataset <a name = "dataset"></a>
The dataset tested with the code is MNIST dataset in CSV format. You can get the dataset from [this](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) kaggle link.

Once you downloaded the dataset you need to extract the dataset and create a folder name data where csv files should be placed. Directory Structure for the data folder should be like this

Data <br>
|___ mnist_train.csv <br>
|___ mnist_test.csv

## Unit Tests <a name = "dataset"></a>
There are some tests defined in the tests folder which can be used to check whether the results of the model are fine are not. To run those tests run the command in following format.

```
python -m unittest tests.[test_name]

# To check the test for network
python -m unittest tests.test_network
```

## Author <a name = "author"></a>

Usama Imdad

- LinkedIn: [Usama Imdad](https://www.linkedin.com/in/usama-imdad/)
- Youtube: [@UsamaImdad](https://www.youtube.com/@UsamaImdad)
- Twitter: [@UsamaImdadSian](https://usamaimdad.cf)
- Website: [https://usamaimdad.cf](https://usamaimdad.cf)

## Show Your Support <a name = "support"></a>
Give a star if this project helped you!