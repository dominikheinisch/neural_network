import numpy as np

from prediction.activation_function import activation_func, activation_func_prim
from loader.mnist_loader import load_data_wrapper
from saver.saver import save

HIDDEN_BIAS = 1


def calc_prediction_accuracy(hidden_weight, output_weight, test_input, test_output):
    net_hidden = test_input @ hidden_weight
    hidden = activation_func(net_hidden)
    hidden_with_bias = np.ones(shape=(hidden.shape[0], hidden.shape[1] + HIDDEN_BIAS))
    hidden_with_bias[:, HIDDEN_BIAS:] = hidden
    net_output = hidden_with_bias @ output_weight
    output = activation_func(net_output)
    res = output
    res = np.argmax(res, axis=1)
    output_res = np.argmax(test_output, axis=1)
    return np.sum(output_res == res) / output_res.shape[0]


def mlp(draw_range):
    tr_zip, va_zip, te_zip = load_data_wrapper("../data")
    tr_in, tr_out = tr_zip

    alpha = 0.015
    images_len = tr_in.shape[0]
    input_data_len = tr_in.shape[1]
    hidden_neurones_size = 50
    output_neurones_size = 10
    weights = []
    weights1 = np.random.uniform(low=-draw_range, high=draw_range, size=(input_data_len * hidden_neurones_size))
    weights1 = np.reshape(weights1, newshape=(input_data_len, hidden_neurones_size))

    hidden_neurones_with_bias_size = hidden_neurones_size + HIDDEN_BIAS
    weights2 = np.random.uniform(low=-draw_range, high=draw_range, size=(hidden_neurones_with_bias_size * output_neurones_size))
    weights2 = np.reshape(weights2, newshape=(hidden_neurones_with_bias_size, output_neurones_size))

    weights.append([weights1, weights2])
    print('result: ', calc_prediction_accuracy(weights1, weights2, *te_zip))

    for j in range(10):
        for i in range(images_len):
            net_hidden = tr_in[i] @ weights1
            hidden = activation_func(net_hidden)
            hidden_with_bias = np.ones(shape=(hidden.shape[0] + HIDDEN_BIAS))
            hidden_with_bias[HIDDEN_BIAS:] = hidden

            net_output = hidden_with_bias @ weights2
            output = activation_func(net_output)

            err_out = (tr_out[i] - output) * activation_func_prim(net_output)
            temp = weights2[HIDDEN_BIAS:] @ err_out
            err_hidden = temp * activation_func_prim(net_hidden)

            weights2_delta = np.transpose(np.tile(hidden_with_bias, reps=(err_out.shape[0], 1))) * (alpha * err_out)
            weights2 = weights2 + weights2_delta

            weights1_delta = np.transpose(np.tile(tr_in[i], reps=(err_hidden.shape[0], 1))) * (alpha * err_hidden)
            weights1 = weights1 + weights1_delta

            if (i + 1) % 1000 == 0:
                print(f'progress print: {i}')
        print(j, ' result: ', calc_prediction_accuracy(weights1, weights2, *te_zip))
        weights.append([weights1, weights2])
    return weights


def mlp_batch(draw_range, batch_size):
    tr_zip, va_zip, te_zip = load_data_wrapper("../data")
    tr_in, tr_out = tr_zip

    alpha = 0.015
    images_len = tr_in.shape[0]
    assert(images_len % batch_size == 0)
    input_data_len = tr_in.shape[1]
    hidden_neurones_size = 50
    output_neurones_size = 10
    weights = []
    weights1 = np.random.uniform(low=-draw_range, high=draw_range, size=(input_data_len * hidden_neurones_size))
    weights1 = np.reshape(weights1, newshape=(input_data_len, hidden_neurones_size))

    hidden_neurones_with_bias_size = hidden_neurones_size + HIDDEN_BIAS
    weights2 = np.random.uniform(low=-draw_range, high=draw_range,
                                 size=(hidden_neurones_with_bias_size * output_neurones_size))
    weights2 = np.reshape(weights2, newshape=(hidden_neurones_with_bias_size, output_neurones_size))

    weights.append([weights1, weights2])
    print('result: ', calc_prediction_accuracy(weights1, weights2, *te_zip))

    batch_indexes = [(i * batch_size, (i + 1) * batch_size) for i in range(images_len // batch_size)]
    for j in range(20):
        for batch_start, batch_end in batch_indexes:
            net_hidden = tr_in[batch_start:batch_end] @ weights1
            hidden = activation_func(net_hidden)
            hidden_with_bias = np.ones(shape=(hidden.shape[0], hidden.shape[1] + HIDDEN_BIAS))
            hidden_with_bias[:, HIDDEN_BIAS:] = hidden

            net_output = hidden_with_bias @ weights2
            output = activation_func(net_output)

            err_out = (tr_out[batch_start:batch_end] - output) * activation_func_prim(net_output)
            temp = err_out @ weights2[HIDDEN_BIAS:, :].transpose()
            err_hidden = temp * activation_func_prim(net_hidden)

            weights2_delta = np.transpose(np.tile(hidden_with_bias,
                                                  reps=(err_out.shape[1], 1, 1))) * err_out
            weights2_delta = np.sum(weights2_delta, axis=1) * (alpha / err_out.shape[0])
            weights2 = weights2 + weights2_delta

            weights1_delta = np.transpose(np.tile(tr_in[batch_start:batch_end],
                                                  reps=(err_hidden.shape[1], 1, 1))) * err_hidden
            weights1_delta = np.sum(weights1_delta, axis=1) * (alpha / err_out.shape[0])
            weights1 = weights1 + weights1_delta

            if batch_start % 1000 == 0:
                print(f'progress print: {batch_start}')
        print(j, ' result: ', calc_prediction_accuracy(weights1, weights2, *te_zip))
        weights.append([weights1, weights2])
    return weights

if __name__ == "__main__":
    # save(data=mlp(draw_range=0.2), filename=f'test_weights_13_bias_{HIDDEN_BIAS}.pkl')

    batch = 1
    save(data=mlp_batch(draw_range=0.2, batch_size=batch), filename=f'test_weights_17_bias_{HIDDEN_BIAS}_batch_{batch}.pkl')
