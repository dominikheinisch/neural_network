import numpy as np

from simulation_type import SimulationType
from file_io import save


def alphas_simulation(func, alphas, w_range, **kwargs):
    results = np.asarray([func(weights=np.random.uniform(-w_range, w_range, 3), alpha=a, **kwargs) for a in alphas])
    epochs = results[:, 1]
    return epochs, alphas


def weight_ranges_simulation(func, alpha, w_ranges, **kwargs):
    results = np.asarray([func(weights=np.random.uniform(-w_r, w_r, 3), alpha=alpha, **kwargs) for w_r in w_ranges])
    epochs = results[:, 1]
    return epochs, w_ranges


def run_many(algo_name, params, n, simulation, **kwargs):
    full_name = f'{simulation.__name__}_for_{algo_name}'
    print(full_name)
    acc = np.zeros(shape=(n,len(params)))
    for i in range(n):
        epochs, params = simulation(**kwargs)
        acc[i] = np.asarray(epochs)
    avg_epochs = np.sum(acc, axis=0) / acc.shape[0]
    print('     params: ', params)
    print('     avg_epochs: ', avg_epochs)
    params_name = simulation.__name__[:simulation.__name__.find('_simulation')]
    err_to_str = f"_err_{kwargs['max_err']}" if algo_name.find('ADALINE') == 0 else ''
    file_name = f"{algo_name}_{params_name}_{'_'.join(map(str, params))}{err_to_str}.json"
    return {'algo_name': algo_name, 'avg_epochs': avg_epochs.tolist(), 'params': params, 'params_name': params_name,
            'max_err': kwargs['max_err'], 'file_name': file_name}


def run_one(algo_name, func, input, output, alpha, w_range, **kwargs):
    print(algo_name)
    weights = np.random.uniform(-w_range, w_range, input.shape[1])
    print(f'random weights: {weights}')
    result, epochs = func(input=input, output=output, alpha=alpha, weights=weights, **kwargs)
    print(f'result weights: {result}, epochs:{epochs}')
    print('input  output  calculated_output')
    for i in range(input.shape[0]):
        print(f'{input[i]}  {output[i]}  {(input[i] @ result):.3f}')


def run_alphas_simulation():
    n=100
    alphas=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    w_range=0.2
    for type in [
        # SimulationType.ADALINE_AND,
        # SimulationType.PERCEPTRON_BIPOLAR_AND,
        # SimulationType.PERCEPTRON_BINARY_AND,
        SimulationType.ADALINE_OR,
        # SimulationType.PERCEPTRON_BIPOLAR_OR,
        # SimulationType.PERCEPTRON_BINARY_OR,
    ]:
        result = run_many(params=alphas, n=n, simulation=alphas_simulation, **type, alphas=alphas,
                          w_range=w_range, max_err=2.5)
        save(f"file_io/data/__{result['file_name']}", result)


def run_weight_ranges_simulation():
    n=100
    w_ranges=[0.0, 0.2, 0.4, 0.6, 0.8, 1]
    alpha=0.01
    for type in [
        SimulationType.ADALINE_OR,
        SimulationType.PERCEPTRON_BIPOLAR_OR,
        SimulationType.PERCEPTRON_BINARY_OR,
    ]:
        result = run_many(params=w_ranges, n=n, simulation=weight_ranges_simulation, **type, alpha=alpha,
                          w_ranges=w_ranges, max_err=2.0)
        save(f"file_io/data/_{result['file_name']}", result)

if __name__== "__main__":
    np.set_printoptions(precision=3)

    # run_one(**SimulationType.PERCEPTRON_BIPOLAR_AND, alpha=0.01, w_range=0.2)
    # run_one(**SimulationType.PERCEPTRON_BIPOLAR_OR, alpha=0.01, w_range=0.2)
    # run_one(**SimulationType.PERCEPTRON_BINARY_AND, alpha=0.01, w_range=0.1)
    # run_one(**SimulationType.PERCEPTRON_BINARY_OR, alpha=0.01, w_range=0.1)
    # run_one(**SimulationType.ADALINE_AND, alpha=0.01, w_range=0.2, max_err=1.2)
    # run_one(**SimulationType.ADALINE_OR, alpha=0.01, w_range=0.2, max_err=1.2)

    run_alphas_simulation()
    # run_weight_ranges_simulation()
