import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def activation_func_prim(x):
    sgm = sigmoid(x)
    return sgm * (1 - sgm)


sigmoid_vectorize = np.vectorize(sigmoid)
sigmoid_prim_vectorize = np.vectorize(activation_func_prim)

activation_func = sigmoid_vectorize
activation_func_prim = np.vectorize(activation_func_prim)
