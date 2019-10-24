import numpy as np

from prediction.network import mlp_batch
from loader.mnist_loader import load_data_wrapper
from loader.loader import load
from saver.saver import save


def get_simul_params(alpha, batch_size, draw_range, epochs, momentum_param):
    return {
        'alpha': alpha,
        'batch_size': batch_size,
        'draw_range': draw_range,
        'epochs': epochs,
        'momentum_param': momentum_param,
    }


def run_once(func, params, **kwargs):
    results = func(**params, **kwargs)
    to_save = {'result': results, **params}
    # save(data=to_save, filename=f'test_weights_22_alpha_{alpha}_batch_{batch_size}_draw_{draw_range}'
    #                             f'momentum_{momentum_param}_epochs_{epochs}.pkl')
    print(to_save)



def run_many(times, func, params, **kwargs):
    acc = []
    for t in range(times):
        acc.append(func(**params, **kwargs)['accuracies'])
        print('simulation iter:', t)
    print(acc)
    avg_accuracy = np.asarray(acc).sum(axis=0) / len(acc)
    print(avg_accuracy)
    return {'times': times, 'avg_accuracy': avg_accuracy, **params}


if __name__ == "__main__":
    loaded_data = load_data_wrapper("../data")
    # params = get_simul_params(alpha=0.01, batch_size=25, draw_range=0.2, epochs=20, momentum_param=0)
    params = get_simul_params(alpha=0.01, batch_size=50, draw_range=0.6, epochs=10, momentum_param=0)
    # # alpha = 0.015 # for batch=1
    # # momentum_param = 0.25
    alpha, batch_size, draw_range, epochs, momentum_param = [params[p] for p in params]
    times = 3
    images_len_divider = 1
    many_res = run_many(times=times, func=mlp_batch, data=loaded_data, params=params,
                        images_len_divider=images_len_divider)
    save(data=many_res, filename=f'simul_alpha_{alpha}_batch_{batch_size}_draw_{draw_range}_momentum_{momentum_param}_'
                                 f'epochs_{epochs}_times_{times}.pkl')
    print(many_res)
    # print(load(filename='simul_alpha_0.007_batch_25_draw_0.2_momentum_0_epochs_2_times_3.pkl'))
    # print(load(filename='simul_alpha_0.01_batch_25_draw_0.2_momentum_0_epochs_20_times_10.pkl'))

    # print(run_once(func=mlp_batch, data=loaded_data, params=params, images_len_divider=100))
