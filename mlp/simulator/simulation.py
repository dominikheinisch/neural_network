import numpy as np

from prediction.network import mlp
from loader.mnist_loader import load_data_wrapper
from saver.saver import save


def get_simul_params(alpha, batch_size, draw_range, hidden_neurones, worse_result_limit, momentum_param):
    return {
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
    return {'times': times, 'avg_epochs': avg_epochs, 'many_results': many_results, **params}


if __name__ == "__main__":
    # # alpha = 0.015 # for batch=1
    # # momentum_param = 0.25
    loaded_data = load_data_wrapper("../data")
    # params = get_simul_params(alpha=0.01, batch_size=25, draw_range=0.2, epochs=20, momentum_param=0)
    params = get_simul_params(alpha=0.04, batch_size=100, draw_range=0.2, hidden_neurones=15,
                              worse_result_limit=2, momentum_param=0)
    alpha, batch_size, draw_range, hidden_neurones, worse_result_limit, momentum_param = [params[p] for p in params]

    # single_res = run_once(func=mlp, data=loaded_data, params=params, images_len_divider=50)
    # print(single_res)
    # save(data=mlp, filename=f'dummy_alpha_{alpha}_batch_{batch_size}_draw_range_{draw_range}_'
    #                         f'epochs_{single_res["epochs"]}_res_{single_res["accuracies"][-1 - worse_result_limit]}_'
    #                         f'momentum_param_{momentum_param}.pkl')

    times = 5
    many_res = run_many(times=times, func=mlp, data=loaded_data, params=params, images_len_divider=1)
    print(many_res)
    save(data=many_res, filename=f'hidden_neurones_simulation_alpha_{alpha}_batch_{batch_size}_draw_range_{draw_range}_'
                                 f'hidden_neurones_{hidden_neurones}_avg_epochs_{many_res["avg_epochs"]}'
                                 f'_times_{times}.pkl')
