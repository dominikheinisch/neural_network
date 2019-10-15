import numpy as np

from adaline import adaline
from consts import *
from simulation import *


# def perceptron(func_name, simulation_func, input, output):
#     print(func_name)
#     weights = np.random.uniform(-1, 1, 3)
#     print(weights)
#     result, epochs = simulation_func(weights=weights, alpha=0.01)
#     print(f'result weights: {result}, epochs:{epochs}')
#     print('input  output  calculated_output')
#     for i in range(input.shape[0]):
#         print(f'{input[i]}  {output[i]}  {input[i] @ result}')


# if __name__== "__main__":
#     # perceptron_binary_and()
#     # perceptron_binary_or()
#     #
#     # perceptron_bipolar_and()
#     # perceptron_bipolar_or()
#
#     # perceptron(func_name='AND', simulation_func=perceptron_binary_func_and, input=INPUT, output=AND_OUTPUT)
#     # perceptron(func_name='OR', simulation_func=perceptron_binary_func_or, input=INPUT, output=OR_OUTPUT)
#     #
#     # perceptron(func_name='AND', simulation_func=perceptron_bipolar_func_and, input=INPUT_BIPOLAR, output=AND_OUTPUT_BIPOLAR)
#     # perceptron(func_name='OR', simulation_func=perceptron_bipolar_func_or, input=INPUT_BIPOLAR, output=OR_OUTPUT_BIPOLAR)
#
#     # adaline(func_name='AND', output=AND_OUTPUT_BIPOLAR)
#     # adaline(func_name='OR', output=OR_OUTPUT_BIPOLAR)
#
#     alpha_simulation()
#
#     # print('AND')
#     # weights = np.random.rand(3)
#     # print(weights)
#     # print(learn_bipolar(input=INPUT_BIPOLAR, output=AND_OUTPUT_BIPOLAR, weights=weights))
#     # print('OR')
#     # weights = np.random.rand(3) * 2 - 1
#     # print(weights)
#     # print(learn_bipolar(input=INPUT_BIPOLAR, output=OR_OUTPUT_BIPOLAR, weights=weights))
#
#     # arr = np.asarray([0, 1, 2, 3])
#     # print(np.random.shuffle(arr))
#     # print(arr)
#     # for i in arr:
#     #     print(i)
