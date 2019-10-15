import numpy as np

def adaline(input, output, weights, alpha, max_err):
    result = np.ones(shape=output.shape)
    is_finished = False
    order = np.arange(input.shape[0])
    epochs = 0
    while not is_finished:
        np.random.shuffle(order)
        is_finished = True
        for i in order:
            result[i] = input[i] @ weights
            delta = output[i] - result[i]
            if(delta * delta > max_err):
                is_finished = False
                weights = weights + 2 * delta * alpha * input[i]
        # print(f'                      weights: {weights}')
        epochs += 1
    return weights, epochs
