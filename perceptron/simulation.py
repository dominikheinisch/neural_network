import numpy as np

from perceptron import perceptron_binary, perceptron_bipolar
from adaline import adaline
from simulation_type import SimulationType

# def perceptron_binary_func_and(weights, alpha):
#     return perceptron_binary(input = INPUT, output = AND_OUTPUT, weights=weights, alpha=alpha)
#
#
# def perceptron_binary_func_or(weights, alpha):
#     return perceptron_binary(input=INPUT, output=OR_OUTPUT, weights=weights, alpha=alpha)
#
#
# def perceptron_bipolar_func_and(weights, alpha):
#     return perceptron_bipolar(input = INPUT_BIPOLAR, output = AND_OUTPUT_BIPOLAR, weights=weights, alpha=alpha)
#
#
# def perceptron_bipolar_func_or(weights, alpha):
#     return perceptron_bipolar(input=INPUT_BIPOLAR, output=OR_OUTPUT_BIPOLAR, weights=weights, alpha=alpha)


def simulation(func_name, simulation_func, input, output):
    print(func_name)
    weights = np.random.uniform(-1, 1, 3)
    print(weights)
    result, epochs = simulation_func(input=input, output=output, weights=weights, alpha=0.01)
    print(f'result weights: {result}, epochs:{epochs}')
    print('input  output  calculated_output')
    for i in range(input.shape[0]):
        print(f'{input[i]}  {output[i]}  {input[i] @ result}')


# def perceptron_binary_and():
#     simulation('AND', simulation_func=perceptron_binary, input=INPUT, output=AND_OUTPUT)
#
#
# def perceptron_binary_or():
#     simulation('OR', simulation_func=perceptron_binary, input=INPUT, output=OR_OUTPUT)
#
#
# def perceptron_bipolar_and():
#     simulation('AND', simulation_func=perceptron_bipolar, input=INPUT_BIPOLAR, output=AND_OUTPUT_BIPOLAR)
#
#
# def perceptron_bipolar_or():
#     simulation('OR', simulation_func=perceptron_bipolar, input=INPUT_BIPOLAR, output=OR_OUTPUT_BIPOLAR)


def alphas_simulation(func, alphas, w_range, **kwargs):
    results = np.asarray([func(weights=np.random.uniform(-w_range, w_range, 3), alpha=a, **kwargs) for a in alphas])
    epochs = results[:, 1]
    return epochs, alphas
    # print([(f'epochs={epochs}, alpha={a}') for (_, epochs), a in zip(results, alphas)])


def weight_ranges_simulation(func, alpha, w_ranges, **kwargs):
    results = np.asarray([func(weights=np.random.uniform(-w_r, w_r, 3), alpha=alpha, **kwargs) for w_r in w_ranges])
    epochs = results[:, 1]
    return epochs, w_ranges
    # print([(f'epochs={epochs}, weight_range=[-{w_r}:{w_r}') for (_, epochs), w_r in zip(results, w_ranges)])


# def adaline(func_name, output):
#     print(func_name)
#     weights = np.random.uniform(-1, 1, 3)
#     print(weights)
#     result, _ = adaline(input=INPUT_BIPOLAR, output=output, weights=weights, alpha=0.01, max_err=0.3)
#     print('result weights:', result)
#     print('input  output  calculated_output')
#     for i in range(INPUT_BIPOLAR.shape[0]):
#         print(f'{INPUT_BIPOLAR[i]}  {output[i]}  {INPUT_BIPOLAR[i] @ result}')


def run_many(name, params, n, simulation, **kwargs):
    print(f'{simulation.__name__} for {name}')
    result = np.zeros(shape=(n,len(params)))
    for i in range(n):
        epochs, params = simulation(**kwargs)
        result[i] = np.asarray(epochs)
    result = np.sum(result, axis=0) / result.shape[0]
    print(params)
    print(result)

if __name__== "__main__":
    # epochs, alphas = alphas_simulation(func=adaline, alphas=[0.001, 0.003, 0.01, 0.03], w_range=0.2)
    # print([(f'epochs={epoch}, alpha={a}') for epoch, a in zip(epochs, alphas)])
    # #
    # epochs, w_ranges = weight_ranges_simulation(func=adaline, alpha=0.01, w_ranges=[0.2, 0.4, 0.6, 0.8, 1])
    # print([(f'epochs={epochs}, weight_range=[-{w_r}:{w_r}]') for epochs, w_r in zip(epochs, w_ranges)])

    n=25
    alphas=[0.001, 0.003, 0.01, 0.03]
    w_range=0.2
    run_many(params=alphas, n=n, simulation=alphas_simulation, **SimulationType.ADALINE_AND, alphas=alphas,
             w_range=w_range, max_err=0.3)
    run_many(params=alphas, n=n, simulation=alphas_simulation, **SimulationType.PERCEPTRON_BIPOLAR_AND, alphas=alphas,
             w_range=w_range)
    run_many(params=alphas, n=n, simulation=alphas_simulation, **SimulationType.PERCEPTRON_BINARY_AND, alphas=alphas,
             w_range=w_range)

    w_ranges=[0.2, 0.4, 0.6, 0.8, 1]
    alpha=0.01
    run_many(params=w_ranges, n=n, simulation=weight_ranges_simulation, **SimulationType.ADALINE_OR,
             alpha=alpha, w_ranges=w_ranges, max_err=0.3)
    run_many(params=w_ranges, n=n, simulation=weight_ranges_simulation, **SimulationType.PERCEPTRON_BIPOLAR_OR,
             alpha=alpha, w_ranges=w_ranges)
    run_many(params=w_ranges, n=n, simulation=weight_ranges_simulation, **SimulationType.PERCEPTRON_BINARY_OR,
             alpha=alpha, w_ranges=w_ranges)