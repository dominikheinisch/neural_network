import math
import numpy as np

from utils.timer import elapsed_timer


def f_relu(matrix):
    # return np.maximum(0, matrix)
    return np.vectorize(lambda x: 1 if x > 0 else 0)(matrix)


def f_relu2(x):
    a = x if x > 0 else -1
    print(a)
    return a

f_relu2 = np.vectorize(f_relu2, otypes=[np.float])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prim(x):
    temp = sigmoid(x)
    return temp * (1 - temp)


def relu2(x):
    np.maximum(x, 0, x)
    return x

def relu(x):
    x[x < 0] = 0
    return x


def sigmoid2(x):
    return np.vectorize(lambda x: 1 / (1 + math.exp(-x)))(x)

def sigmoid_prim2(x):
    s = sigmoid2(x)
    return s * (1 - s)

if __name__ == "__main__":
    # with elapsed_timer() as timer:
    #     n = 10000
    #     matrix = np.arange(n) / n  - 0.5
    #     print(f'timer: {timer():.6f}')
    #     relu1 = f_relu(matrix)
    #     # print('relu1', relu1)
    #     print(f'timer: {timer():.6f}')
    #     relu2 = np.copy(matrix)
    #     print(f'timer: {timer():.6f}')
    #     relu2[relu2 < 0] = 0
    #     print(f'timer: {timer():.6f}')
    #
    #     print('matrix', matrix)
    #     print('relu1', relu1)
    #     print('relu2', relu2)
    #     print(sum(relu1))
    #     print(sum(relu2))

    n = 10
    matrix = np.arange(n) / n - 0.5
    with elapsed_timer() as timer:
        for i in range(11):
            res1 = relu(matrix)
        print(i, f'timer: {timer():.6f}')
        # print('sigmoid_prim', sigmoid_prim)
        # si =

    with elapsed_timer() as timer:
        for i in range(11):
            res2 = relu2(matrix)
        print(i, f'timer: {timer():.6f}')

    with elapsed_timer() as timer:
        for i in range(2):
            res3 = f_relu2(matrix)
        print(i, f'timer: {timer():.6f}')

    print(res3)
    print(np.all(res1 == res2))
    print(np.all(res1 == res3))
        # print('matrix', matrix)
        # print(sum(sigmoid))
