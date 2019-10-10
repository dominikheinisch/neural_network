import numpy as np
from time import sleep

from perceptron import learn, learn_bipolar
from adaline import learn_adaline


INPUT = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
AND_OUTPUT = np.asarray([[0], [0], [0], [1]])
OR_OUTPUT = np.asarray([[0], [1], [1], [1]])


ADALINE_INPUT = np.asarray([[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
ADALINE_AND_OUTPUT = np.asarray([[-1], [-1], [-1], [1]])
ADALINE_OR_OUTPUT = np.asarray([[-1], [1], [1], [1]])


INPUT_BIPOLAR = ADALINE_INPUT
AND_OUTPUT_BIPOLAR = ADALINE_AND_OUTPUT
OR_OUTPUT_BIPOLAR = ADALINE_OR_OUTPUT

if __name__== "__main__":
    print('AND')
    weights = np.random.rand(3)
    print(weights)
    result = learn_adaline(input=ADALINE_INPUT, output=ADALINE_AND_OUTPUT, weights=weights)
    print('result weights:', result)
    for row in range(ADALINE_INPUT.shape[0]):
        print(ADALINE_INPUT[row], '   ', ADALINE_INPUT[row] @ result, ADALINE_AND_OUTPUT[row])
    print('OR')
    weights = np.random.rand(3)
    print(weights)
    result = learn_adaline(input=ADALINE_INPUT, output=ADALINE_OR_OUTPUT, weights=weights)
    print('result weights:', result)
    for row in range(ADALINE_INPUT.shape[0]):
        print(ADALINE_INPUT[row], '   ', ADALINE_INPUT[row] @ result, ADALINE_OR_OUTPUT[row])

    # print('AND')
    # weights = np.random.rand(2)
    # print(weights)
    # print(learn(input=INPUT, output=AND_OUTPUT, weights=weights))
    # print('OR')
    # weights = np.random.rand(2)
    # print(weights)
    # print(learn(input=INPUT, output=OR_OUTPUT, weights=weights))

    # print('AND')
    # weights = np.random.rand(3)
    # print(weights)
    # print(learn_bipolar(input=INPUT_BIPOLAR, output=AND_OUTPUT_BIPOLAR, weights=weights))
    # print('OR')
    # weights = np.random.rand(3) * 2 - 1
    # print(weights)
    # print(learn_bipolar(input=INPUT_BIPOLAR, output=OR_OUTPUT_BIPOLAR, weights=weights))

    # arr = np.asarray([0, 1, 2, 3])
    # print(np.random.shuffle(arr))
    # print(arr)
    # for i in arr:
    #     print(i)