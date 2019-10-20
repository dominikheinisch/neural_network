import numpy as np

from activation_function import activation_func, activation_func_prim
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


if __name__ == "__main__":
    tr_zip, va_zip, te_zip = load_data_wrapper("../data")
    tr_in, tr_out = tr_zip

    alpha = 0.01
    images_len = tr_in.shape[0]
    input_data_len = tr_in.shape[1]
    hidden_neurones_size = 50
    output_neurones_size = 10
    weights = []
    weights1 = np.random.uniform(low=-1, high=1, size=(input_data_len * hidden_neurones_size))
    weights1 = np.reshape(weights1, newshape=(input_data_len, hidden_neurones_size))

    hidden_neurones_with_bias_size = hidden_neurones_size + HIDDEN_BIAS
    weights2 = np.random.uniform(low=-1, high=1, size=(hidden_neurones_with_bias_size * output_neurones_size))
    weights2 = np.reshape(weights2, newshape=(hidden_neurones_with_bias_size, output_neurones_size))

    weights.append([weights1, weights2])
    print('result: ', calc_prediction_accuracy(weights1, weights2, *te_zip))
    # print(weights1.shape)
    # print(tr_in.shape)
    # print(tr_out.shape)

    for j in range(1):
        for i in range(10000):
            net_hidden = tr_in[i] @ weights1
            hidden = activation_func(net_hidden)
            hidden_with_bias = np.ones(shape=(hidden.shape[0] + HIDDEN_BIAS))
            hidden_with_bias[HIDDEN_BIAS:] = hidden
            # print(hidden)

            net_output = hidden_with_bias @ weights2
            output = activation_func(net_output)
            # print(output.shape)
            # print('output', output, tr_out[i])

            # print('--------------------------------')
            # print(output - tr_out[i])
            # print('--------------------------------activation_func_prim')
            # print(activation_func_prim(net_output))
            # print('--------------------------------')
            err_out = (tr_out[i] - output) * activation_func_prim(net_output)
            # print('temp:')
            temp = weights2[HIDDEN_BIAS:] @ err_out
            # print(temp)
            # print(activation_func_prim(net_hidden))
            err_hidden = temp * activation_func_prim(net_hidden)
            # print('weights2', weights2.shape)
            # print('err_out', err_out.shape)
            # print('hidden', hidden.shape)

            weights2_delta = np.transpose(np.tile(hidden_with_bias, reps=(err_out.shape[0], 1))) * (alpha * err_out)
            weights2 = weights2 + weights2_delta

            weights1_delta = np.transpose(np.tile(tr_in[i] * alpha, reps=(err_hidden.shape[0], 1))) * (alpha * err_hidden)
            weights1 = weights1 + weights1_delta

            # print('delta2', weights2_delta.shape)
            # print(weights2_delta)
            # print(weights2)
            # print(weights1)
            if (i + 1) % 1000 == 0:
                print(i)
        print(j, ' result: ', calc_prediction_accuracy(weights1, weights2, *te_zip))
        weights.append([weights1, weights2])
    save(data=weights, filename=f'test_weights_8_bias_{HIDDEN_BIAS}.pkl')
