import numpy as np

def learn(input, output, weights, alpha, result_func):
    result = np.ones(shape=output.shape)
    order = np.arange(input.shape[0])
    epochs = 0
    is_finished = False
    while not is_finished:
        np.random.shuffle(order)
        is_finished = True
        for i in order:
            result[i] = input[i] @ weights
            result[i] = np.vectorize(result_func)(result[i])
            delta = output[i] - result[i]
            if(not delta == 0):
                is_finished = False
                weights = weights + delta * alpha * input[i]
        epochs += 1
    return weights, epochs


def binary_result(elem):
    return 1 if elem > 0 else 0


def bipolar_result(elem):
    return 1 if elem > 0 else -1


def perceptron_binary(input, output, weights, alpha):
    return learn(input, output, weights, alpha, result_func=binary_result)


def perceptron_bipolar(input, output, weights, alpha):
    return learn(input, output, weights, alpha, result_func=bipolar_result)
