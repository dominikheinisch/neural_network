import numpy as np

from loader.loader import load
from loader.mnist_loader import load_data_wrapper
from prediction.network import calc_prediction_accuracy, mlp_batch

def test_calc_prediction_accuracy():
    weights = load(filename='test_weights.pkl')
    _, _, test_data = load_data_wrapper("../data")
    te_in, te_out = test_data
    random_weights = weights[0]
    assert(calc_prediction_accuracy(*random_weights, te_in, te_out) == 0.0993)
    calculated_weights = weights[3]
    assert(calc_prediction_accuracy(*calculated_weights, te_in, te_out) == 0.7478)


def test_mlp():
    loaded_weights = load(filename='test_mlp_weights_bias_1_batch_1.pkl')
    np.random.seed(0)
    weights = mlp_batch(draw_range=0.2, batch_size=1, epochs=1, images_len_divider=10)
    for (w11 , w12), (w21, w22) in zip(loaded_weights, weights):
        assert(np.allclose(a=w11, b=w21))
        assert (np.allclose(a=w12, b=w22))


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


if __name__ == "__main__":
    test_calc_prediction_accuracy()
    test_hidden_backprop()
    test_hidden_backprop_with_batch()
    test_hidden_backprop_with_batch_2()
    test_mlp()
