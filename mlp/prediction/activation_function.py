import numpy as np
from collections import namedtuple

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


Activation = namedtuple('Activation', ['activation', 'activation_prim'])


SIGMOID = Activation(activation=sigmoid, activation_prim=sigmoid_prim)
RELU = Activation(activation=relu, activation_prim=relu_prim)
