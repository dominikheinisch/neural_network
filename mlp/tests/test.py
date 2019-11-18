import numpy as np

from plotters.chart_plotter import calc_avg_duration
from loader.loader import load
from loader.mnist_loader import load_data_wrapper
from prediction.activation_function import SIGMOID, RELU
from prediction.network import add_bias, calc_prediction_accuracy, mlp

def test_calc_prediction_accuracy():
    loaded_weights = load(filename='test_once_sigmoid_alpha_0.04_batch_100_draw_range_0.001_hidden_neurones_50_'
                                   'test_accuracy_0.968.pkl')['weights']
    _, _, test_data = load_data_wrapper("../data")
    test_data = add_bias(data=test_data)
    te_in, te_out = test_data
    weights_tested = loaded_weights[0]
    assert(calc_prediction_accuracy(SIGMOID.activation, *weights_tested, te_in, te_out) == 0.101)
    weights_tested = loaded_weights[-1]
    assert(calc_prediction_accuracy(SIGMOID.activation, *weights_tested, te_in, te_out) == 0.9687)


def test_mlp():
    loaded_data = load_data_wrapper("../data")
    loaded_result = load(filename='test_once_sigmoid_alpha_0.05_batch_100_draw_range_0.2_hidden_neurones_15_mom_0.5_'
                                   'accuracy_0.7321.pkl')
    np.random.seed(0)
    result = mlp(data=loaded_data, activation=SIGMOID, alpha=0.05, batch_size=100, draw_range=0.2,
                 hidden_neurones=15, worse_result_limit=1, momentum_param=0.5, is_adagrad=False, images_len_divider=250)
    for (w11 , w12), (w21, w22) in zip(loaded_result['weights'], result['weights']):
        assert(np.allclose(a=w11, b=w21))
        assert(np.allclose(a=w12, b=w22))
    assert (len(loaded_result['weights']) == len(result['weights']))


def test_hidden_backprop():
    hidden_with_bias = np.asarray(list(range(5))) + 1
    err_out = np.asarray(list(range(3))) + 1
    alpha = 1
    weights2_delta = np.transpose(np.tile(hidden_with_bias, reps=(err_out.shape[0], 1))) * (alpha * err_out)
    result = np.asarray([
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9],
        [4, 8, 12],
        [5, 10, 15],
    ])
    assert (np.all(result == weights2_delta))


def hidden_backprop_with_batch(hidden_with_bias, err_out, alpha):
    temp = np.tile(hidden_with_bias, reps=(err_out.shape[1], 1, 1))
    weights2_delta = np.transpose(temp) * err_out
    return np.sum(weights2_delta, axis=1) * (alpha / err_out.shape[0])


def test_hidden_backprop_with_batch():
    hidden_with_bias = np.asarray([list(range(5)), [1] * 5]) * 2
    err_out = np.asarray([list(range(3)), [1] * 3]) * 2
    alpha = 1
    weights2_delta = hidden_backprop_with_batch(hidden_with_bias, err_out, alpha)
    result = np.asarray([
        [ 2,  2,  2],
        [ 2,  4,  6],
        [ 2,  6, 10],
        [ 2,  8, 14],
        [ 2, 10, 18],
    ])
    assert (np.all(result == weights2_delta))


def test_hidden_backprop_with_batch_2():
    hidden_with_bias = np.asarray([list(range(5)), [1] * 5]) * 2
    err_out = np.asarray([[0, 2, 4], [1, 2, 3]])
    alpha = 1
    weights2_delta = hidden_backprop_with_batch(hidden_with_bias, err_out, alpha)
    result = np.asarray([
        [ 1,  2,  3],
        [ 1,  4,  7],
        [ 1,  6, 11],
        [ 1,  8, 15],
        [ 1, 10, 19],
    ])
    assert (np.all(result == weights2_delta))


def test_calc_avg_duration():
    input = [['1', '2', '4'], ['1', '2', '3']]
    assert(np.all([1, 2, 3.5] == calc_avg_duration(input)))
    input = [['1', '2'], ['2', '4', '6', '11'], ['6']]
    assert(np.all([3, 6, 9, 13] == calc_avg_duration(input)))


if __name__ == "__main__":
    test_calc_prediction_accuracy()
    test_mlp()
    test_hidden_backprop()
    test_hidden_backprop_with_batch()
    test_hidden_backprop_with_batch_2()
    test_calc_avg_duration()
