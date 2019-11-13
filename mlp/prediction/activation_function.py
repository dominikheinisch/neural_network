import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prim(x):
    sgm = sigmoid(x)
    return sgm * (1 - sgm)


def relu(x):
    x[x < 0] = 0
    return x


def relu_prim(x):
    return x > 0


SIGMOID = sigmoid, sigmoid_prim
RELU = relu, relu_prim
