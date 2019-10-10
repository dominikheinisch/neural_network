import numpy as np

def learn(input, output, weights, treshold = 0.5, alpha = 0.02):
    result = np.ones(shape=output.shape)
    is_finished = False;
    while not is_finished:
        is_finished = True
        for i in range(input.shape[0]):
            result[i] = input[i] @ weights
            # result[i] = 1 if result[i] > treshold else 0
            # TODO check efficiency
            result[i] = np.vectorize(calc_result)(result[i], treshold)
            delta = output[i] - result[i]
            if(not delta == 0):
                is_finished = False
                weights = weights + delta * alpha * input[i]
    return weights


def calc_result(data, treshold):
    return 1 if data > treshold else 0


def learn_bipolar(input, output, weights, treshold = 0.0, alpha = 0.02):
    result = np.ones(shape=output.shape)
    is_finished = False
    arr = np.asarray([0, 1, 2, 3])
    while not is_finished:
        np.random.shuffle(arr)
        is_finished = True
        for i in range(input.shape[0]):
            result[i] = input[i] @ weights
            result[i] = 1 if result[i] > treshold else -1
            delta = output[i] - result[i]
            if(not delta == 0):
                is_finished = False
                weights = weights + delta * alpha * input[i]
        print('                      weights:', weights)
    return weights
