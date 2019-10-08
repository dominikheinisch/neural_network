import numpy as np
from time import sleep


def get_random():
    return np.random.rand(2)


INPUT = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
AND_OUTPUT = np.asarray([[0], [0], [0], [1]])
OR_OUTPUT = np.asarray([[0], [1], [1], [1]])


def fun(output, treshold = 0.5, alpha = 0.05):
    w_m = get_random()
    print(w_m)
    result = np.ones(shape=(4, 1))
    for i in range(INPUT.shape[0]):
        result[i] = INPUT[i] @ w_m
        result[i] = 1 if result[i] > treshold else 0


    is_finished = False;
    while not is_finished:
        is_finished = True
        for i in range(INPUT.shape[0]):
            delta = output[i] - result[i]
            print(i, delta)
            if(not delta == 0):
                print([i] * 10)
                is_finished = False
                w_m = w_m + alpha * w_m * delta
            for i in range(INPUT.shape[0]):
                result[i] = INPUT[i] @ w_m
                result[i] = 1 if result[i] > treshold else 0
        print(w_m)
        print(result)
        # sleep(1.5)
    return w_m#, result

print(fun(output=AND_OUTPUT))
# print(fun(output=OR_OUTPUT))

