import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_prim(x):
    sgm = sigmoid(x)
    return sgm * (1 - sgm)


def relu(x):
    return max(x, 0)


def relu_prim(x):
    return 1 if x > 0 else 0


SIGMOID = sigmoid, sigmoid_prim
RELU = relu, relu_prim
