import numpy as np
from time import sleep

from perceptron import learn


INPUT = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
AND_OUTPUT = np.asarray([[0], [0], [0], [1]])
OR_OUTPUT = np.asarray([[0], [1], [1], [1]])



if __name__== "__main__":
    print('AND')
    weights = np.random.rand(2)
    print(weights)
    print(learn(input=INPUT, output=AND_OUTPUT, weights=weights))

    print('OR')
    weights = np.random.rand(2)
    print(weights)
    print(learn(input=INPUT, output=OR_OUTPUT, weights=weights))

