import numpy as np

from prediction.activation_function import SIGMOID, RELU
from prediction.network import mlp
from loader.mnist_loader import load_data_wrapper
from saver.saver import save


def get_simul_params(activation, alpha, batch_size, draw_range, hidden_neurones, worse_result_limit, momentum_param):
    return {
        'activation': activation,
        'alpha': alpha,
        'batch_size': batch_size,
        'draw_range': draw_range,
        'hidden_neurones': hidden_neurones,
        'worse_result_limit': worse_result_limit,
        'momentum_param': momentum_param,
    }


def run_once(func, params, **kwargs):
    results = func(**params, **kwargs)
    return {**results, **params}


def run_many(times, func, params, **kwargs):
    many_results = []
    for t in range(times):
        single_result = func(**params, **kwargs)
        del single_result['weights']
        many_results.append(single_result)
        print('simulation iter:', t)
    print(many_results)
    avg_epochs = sum([res['epochs'] for res in many_results]) / len(many_results)
    params['activation'] = params['activation'][0].__name__
    return {'times': times, 'avg_epochs': avg_epochs, 'many_results': many_results, **params}


def run_simulation(name, data, times, params):
    _, alpha, batch_size, draw_range, hidden_neurones, worse_result_limit, momentum_param = (params[p] for p in params)
    many_res = run_many(times=times, func=mlp, data=data, params=params, images_len_divider=1)
    print(many_res)
    save(data=many_res, filename=f'{name}_simulation_{many_res["activation"]}_alpha_{alpha}_batch_{batch_size}_'
                                 f'draw_range_{draw_range}_hidden_neurones_{hidden_neurones}_'
                                 f'avg_epochs_{many_res["avg_epochs"]}_times_{times}.pkl')

if __name__ == "__main__":
    loaded_data = load_data_wrapper("../data")
    run_simulation(name='batch_simulation', data=loaded_data, times=5,
                   params=get_simul_params(activation=SIGMOID, alpha=0.01, batch_size=10, draw_range=0.2,
                                           hidden_neurones=50, worse_result_limit=5, momentum_param=0))

    # # single_res = run_once(func=mlp, data=loaded_data, params=params, images_len_divider=50)
    # # print(single_res)
    # # save(data=mlp, filename=f'dummy_alpha_{alpha}_batch_{batch_size}_draw_range_{draw_range}_'
    # #                         f'epochs_{single_res["epochs"]}_res_{single_res["accuracies"][-1 - worse_result_limit]}_'
    # #                         f'momentum_param_{momentum_param}.pkl')
