import numpy as np

def learn_adaline(input, output, weights, alpha = 0.01, max_err=0.3):
    result = np.ones(shape=output.shape)
    is_finished = False
    arr = np.asarray([0, 1, 2, 3])
    while not is_finished:
        np.random.shuffle(arr)
        is_finished = True
        for i in range(input.shape[0]):
            result[i] = input[i] @ weights
            delta = output[i] - result[i]
            if(delta * delta > max_err):
                is_finished = False
                weights = weights + 2 * delta * alpha * input[i]
        print('                      weights:', weights)
    return weights


def calc_result(data, treshold):
    return 1 if data > treshold else 0
