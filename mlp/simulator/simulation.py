import numpy as np

from prediction.activation_function import SIGMOID, RELU
from prediction.network import mlp
from loader.mnist_loader import load_data_wrapper
from saver.saver import save


def get_simul_params(activation, alpha, batch_size, draw_range, hidden_neurones, worse_result_limit, momentum_param=0,
                     is_adagrad=False):
    return {
        'activation': activation,
        'alpha': alpha,
        'batch_size': batch_size,
        'draw_range': draw_range,
        'hidden_neurones': hidden_neurones,
        'worse_result_limit': worse_result_limit,
        'momentum_param': momentum_param,
        'is_adagrad': is_adagrad,
    }


def run_many(repeats, func, params, **kwargs):
    many_results = []
    for t in range(repeats):
        single_result = func(**params, **kwargs)
        del single_result['weights']
        many_results.append(single_result)
        print('simulation iter:', t)
    print(many_results)
    avg_epochs = sum([res['epochs'] for res in many_results]) / len(many_results)
    params['activation'] = params['activation'].activation.__name__
    return {'repeats': repeats, 'avg_epochs': avg_epochs, 'many_results': many_results, **params}


def run_simulation(name, data, repeats, params):
    _, alpha, batch_size, draw_range, hidden_neurones, worse_result_limit, momentum_param, is_adagrad\
        = (params[p] for p in params)
    many_res = run_many(repeats=repeats, func=mlp, data=data, params=params, images_len_divider=1)
    print(many_res)
    save(data=many_res, filename=f'{name}_simul_{many_res["activation"]}_alpha_{alpha}_batch_{batch_size}_'
                                 f'draw_r_{draw_range}_hidden_n_{hidden_neurones}_mom_{momentum_param}_'
                                 f'avg_epochs_{many_res["avg_epochs"]}_reps_{repeats}.pkl')


def run_once(name, params, **kwargs):
    _, alpha, batch_size, draw_range, hidden_neurones, worse_result_limit, momentum_param, is_adagrad \
        = (params[p] for p in params)
    result = mlp(**params, **kwargs, images_len_divider=250)
    print(result)
    save(data=result, filename=f'{name}_once_{params["activation"].activation.__name__}_alpha_{alpha}_'
                               f'batch_{batch_size}_draw_range_{draw_range}_hidden_neurones_{hidden_neurones}_'
                               f'mom_{momentum_param}_accuracy_{result["test_accuracies"][-1 - worse_result_limit]}.pkl')


if __name__ == "__main__":
    loaded_data = load_data_wrapper("../data")

    np.random.seed(0)
    # run_once(name='', data=loaded_data,
    #          params=get_simul_params(activation=RELU, alpha=0.01, batch_size=10, draw_range=0.0001,
    #                                  hidden_neurones=25, worse_result_limit=3, momentum_param=0.5))
    run_once(name='', data=loaded_data,
             params=get_simul_params(activation=SIGMOID, alpha=0.05, batch_size=100, draw_range=0.2,
                                     hidden_neurones=15, worse_result_limit=1, momentum_param=0.5, is_adagrad=False))

    # run_simulation(name='alpha2', data=loaded_data, repeats=1,
    #                params=get_simul_params(activation=RELU, alpha=0.003, batch_size=10, draw_range=0.2,
    #                                        hidden_neurones=50, worse_result_limit=2, momentum_param=0))
    # for batch_size in [5, 10, 25, 50]:
    #     run_simulation(name='batch2', data=loaded_data, repeats=2,
    #                    params=get_simul_params(activation=RELU, alpha=0.01, batch_size=batch_size, draw_range=0.01,
    #                                            hidden_neurones=50, worse_result_limit=2, momentum_param=0))

    # for momentum in [0, 0.25, 0.5, 0.75, 1.0]:
    #     run_simulation(name='momentum2', data=loaded_data, repeats=5,
    #                    params=get_simul_params(activation=SIGMOID, alpha=0.01, batch_size=100, draw_range=0.2,
    #                                            hidden_neurones=25, worse_result_limit=2, momentum_param=momentum))
    # for hidden_neurones in [15, 25, 50, 75]:
    #     run_simulation(name='hidden_neurones4', data=loaded_data, repeats=5,
    #                    params=get_simul_params(activation=RELU, alpha=0.01, batch_size=5, draw_range=0.2,
    #                                            hidden_neurones=hidden_neurones, worse_result_limit=5, momentum_param=0))

    #
    # for alpha in [0.08, 0.02, 0.01, 0.005]:
    #     run_simulation(name='alpha2', data=loaded_data, repeats=5,
    #                    params=get_simul_params(activation=SIGMOID, alpha=alpha, batch_size=100, draw_range=0.2,
    #                                            hidden_neurones=50, worse_result_limit=2, momentum_param=0))
    #
    # for draw_range in [0.4, 0.6, 0.8, 1.0]:
    #     run_simulation(name='draw_range2', data=loaded_data, repeats=5,
    #                    params=get_simul_params(activation=SIGMOID, alpha=0.04, batch_size=100, draw_range=draw_range,
    #                                            hidden_neurones=50, worse_result_limit=2, momentum_param=0))
