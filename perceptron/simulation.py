import numpy as np

from simulation_type import SimulationType


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


def run_many(name, params, n, simulation, **kwargs):
    print(f'{simulation.__name__} for {name}')
    result = np.zeros(shape=(n,len(params)))
    for i in range(n):
        epochs, params = simulation(**kwargs)
        result[i] = np.asarray(epochs)
    result = np.sum(result, axis=0) / result.shape[0]
    print(params)
    print(result)


def run_one(name, func, input, output, alpha, w_range, **kwargs):
    print(name)
    weights = np.random.uniform(-w_range, w_range, 3)
    print(f'random weights: {weights}')
    result, epochs = func(input=input, output=output, alpha=alpha, weights=weights, **kwargs)
    print(f'result weights: {result}, epochs:{epochs}')
    print('input  output  calculated_output')
    for i in range(input.shape[0]):
        print(f'{input[i]}  {output[i]}  {input[i] @ result}')


if __name__== "__main__":
    run_one(**SimulationType.PERCEPTRON_BIPOLAR_AND, alpha=0.01, w_range=0.2)
    run_one(**SimulationType.PERCEPTRON_BIPOLAR_OR, alpha=0.01, w_range=0.2)
    run_one(**SimulationType.PERCEPTRON_BINARY_AND, alpha=0.01, w_range=0.2)
    run_one(**SimulationType.PERCEPTRON_BINARY_OR, alpha=0.01, w_range=0.2)
    run_one(**SimulationType.ADALINE_AND, alpha=0.01, w_range=0.2, max_err=0.3)
    run_one(**SimulationType.ADALINE_OR, alpha=0.01, w_range=0.2, max_err=0.3)

    # n=25
    # alphas=[0.001, 0.003, 0.01, 0.03]
    # w_range=0.2
    # run_many(params=alphas, n=n, simulation=alphas_simulation, **SimulationType.ADALINE_AND, alphas=alphas,
    #          w_range=w_range, max_err=0.3)
    # run_many(params=alphas, n=n, simulation=alphas_simulation, **SimulationType.PERCEPTRON_BIPOLAR_AND, alphas=alphas,
    #          w_range=w_range)
    # run_many(params=alphas, n=n, simulation=alphas_simulation, **SimulationType.PERCEPTRON_BINARY_AND, alphas=alphas,
    #          w_range=w_range)
    #
    # w_ranges=[0.2, 0.4, 0.6, 0.8, 1]
    # alpha=0.01
    # run_many(params=w_ranges, n=n, simulation=weight_ranges_simulation, **SimulationType.ADALINE_OR,
    #          alpha=alpha, w_ranges=w_ranges, max_err=0.3)
    # run_many(params=w_ranges, n=n, simulation=weight_ranges_simulation, **SimulationType.PERCEPTRON_BIPOLAR_OR,
    #          alpha=alpha, w_ranges=w_ranges)
    # run_many(params=w_ranges, n=n, simulation=weight_ranges_simulation, **SimulationType.PERCEPTRON_BINARY_OR,
    #          alpha=alpha, w_ranges=w_ranges)