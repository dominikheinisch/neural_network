import math
import numpy as np

from loader.mnist_loader import load_data, load_data_wrapper, vectorize_results
from saver.saver import save


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def activation_func_prim(x):
    sgm = sigmoid(x)
    return sgm * (1 - sgm)


sigmoid_vectorize = np.vectorize(sigmoid)
activation_func = sigmoid_vectorize
activation_func_prim = np.vectorize(activation_func_prim)


def check_result(hidden_weight, output_weight, test_input, test_output):
    net_hidden = test_input @ hidden_weight
    hidden = activation_func(net_hidden)
    net_output = hidden @ output_weight
    output = activation_func(net_output)
    res = output
    res = np.argmax(res, axis=1)
    output_res = np.argmax(test_output, axis=1)
    return np.sum(output_res == res) / output_res.shape[0]


if __name__ == "__main__":
    # tr, va, te = load_data("../data")
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

    weights2 = np.random.uniform(low=-1, high=1, size=(hidden_neurones_size * output_neurones_size))
    weights2 = np.reshape(weights2, newshape=(hidden_neurones_size, output_neurones_size))

    weights.append([weights1, weights2])
    print('result: ', check_result(weights1, weights2, *te_zip))
    # print(weights1.shape)
    # print(tr_in.shape)
    # print(tr_out.shape)

    for j in range(3):
        for i in range(50000):
            net_hidden = tr_in[i] @ weights1
            hidden = activation_func(net_hidden)
            # print(hidden.shape)
            # print(hidden)

            net_output = hidden @ weights2
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
            temp = weights2 @ err_out
            # print(temp)
            # print(activation_func_prim(net_hidden))
            err_hidden = temp * activation_func_prim(net_hidden)
            # print('weights2', weights2.shape)
            # print('err_out', err_out.shape)
            # print('hidden', hidden.shape)

            weights2_delta = np.transpose(np.tile(hidden, reps=(err_out.shape[0], 1))) * (alpha * err_out)
            weights2 = weights2 + weights2_delta

            weights1_delta = np.transpose(np.tile(tr_in[i] * alpha, reps=(err_hidden.shape[0], 1))) * (alpha * err_hidden)
            weights1 = weights1 + weights1_delta

            # print('delta2', weights2_delta.shape)
            # print(weights2_delta)
            # print(weights2)
            # print(weights1)
            if (i + 1) % 1000 == 0:
                print(i)
        print(j, ' result: ', check_result(weights1, weights2, *te_zip))
        weights.append([weights1, weights2])
    save(data=weights, filename='test_weights_2.pkl')
