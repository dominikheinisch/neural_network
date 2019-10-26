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


def mlp(data, alpha, draw_range, batch_size, epochs, momentum_param=0, images_len_divider=1):
    training_data, validation_zip, test_data = data
    tr_in, tr_out = training_data

    images_len = tr_in.shape[0]
    assert(images_len % batch_size == 0)
    input_data_len = tr_in.shape[1]
    hidden_neurones_size = 50
    output_neurones_size = 10
    weights = []
    accuracies = []
    weights1 = np.random.uniform(low=-draw_range, high=draw_range, size=(input_data_len * hidden_neurones_size))
    weights1 = np.reshape(weights1, newshape=(input_data_len, hidden_neurones_size))

    hidden_neurones_with_bias_size = hidden_neurones_size + HIDDEN_BIAS
    weights2 = np.random.uniform(low=-draw_range, high=draw_range,
                                 size=(hidden_neurones_with_bias_size * output_neurones_size))
    weights2 = np.reshape(weights2, newshape=(hidden_neurones_with_bias_size, output_neurones_size))

    weights.append([weights1, weights2])
    accuracies.append(calc_prediction_accuracy(weights1, weights2, *test_data))
    print('result: ', accuracies[-1])

    batch_indexes = [(i * batch_size, (i + 1) * batch_size) for i in range(images_len //
                                                                           images_len_divider // batch_size)]
    for j in range(epochs):
        weights1_delta_prev = 0
        weights2_delta_prev = 0
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
            weights2_delta = np.sum(weights2_delta, axis=1) * alpha
            weights2 = weights2 + weights2_delta + weights2_delta_prev  * momentum_param
            weights2_delta_prev = weights2_delta

            weights1_delta = np.transpose(np.tile(tr_in[batch_start:batch_end],
                                                  reps=(err_hidden.shape[1], 1, 1))) * err_hidden
            weights1_delta = np.sum(weights1_delta, axis=1) * alpha
            weights1 = weights1 + weights1_delta + weights1_delta_prev * momentum_param
            weights1_delta_prev = weights1_delta

            if batch_start % 1000 == 0:
                print(f'progress print: {batch_start}')
        weights.append([weights1, weights2])
        accuracies.append(calc_prediction_accuracy(weights1, weights2, *test_data))
        print(j, ' result: ', accuracies[-1])
    return {'weights': weights[-1], 'accuracies': accuracies}
    # return weights


def mlp_batch(data, alpha, draw_range, batch_size, epochs, momentum_param=0, images_len_divider=1):
    training_data, validation_zip, test_data = data
    tr_in, tr_out = training_data

    images_len = tr_in.shape[0]
    assert(images_len % batch_size == 0)
    input_data_len = tr_in.shape[1]
    hidden_neurones_size = 50
    output_neurones_size = 10
    weights = []
    accuracies = []
    weights1 = np.random.uniform(low=-draw_range, high=draw_range, size=(input_data_len * hidden_neurones_size))
    weights1 = np.reshape(weights1, newshape=(input_data_len, hidden_neurones_size))

    hidden_neurones_with_bias_size = hidden_neurones_size + HIDDEN_BIAS
    weights2 = np.random.uniform(low=-draw_range, high=draw_range,
                                 size=(hidden_neurones_with_bias_size * output_neurones_size))
    weights2 = np.reshape(weights2, newshape=(hidden_neurones_with_bias_size, output_neurones_size))

    weights.append([weights1, weights2])
    accuracies.append(calc_prediction_accuracy(weights1, weights2, *test_data))
    print('result: ', accuracies[-1])

    batch_indexes = [(i * batch_size, (i + 1) * batch_size) for i in range(images_len //
                                                                           images_len_divider // batch_size)]
    for j in range(epochs):
        weights1_delta_prev = 0
        weights2_delta_prev = 0
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
            weights2_delta = np.sum(weights2_delta, axis=1) * alpha
            weights2 = weights2 + weights2_delta + weights2_delta_prev  * momentum_param
            weights2_delta_prev = weights2_delta

            weights1_delta = np.transpose(np.tile(tr_in[batch_start:batch_end],
                                                  reps=(err_hidden.shape[1], 1, 1))) * err_hidden
            weights1_delta = np.sum(weights1_delta, axis=1) * alpha
            weights1 = weights1 + weights1_delta + weights1_delta_prev * momentum_param
            weights1_delta_prev = weights1_delta

            if batch_start % 1000 == 0:
                print(f'progress print: {batch_start}')
        weights.append([weights1, weights2])
        accuracies.append(calc_prediction_accuracy(weights1, weights2, *test_data))
        print(j, ' result: ', accuracies[-1])
    return {'weights': weights[-1], 'accuracies': accuracies}
    # return weights

if __name__ == "__main__":
    loaded_data = load_data_wrapper("../data")

    # alpha = 0.015 # for batch=1
    alpha = 0.007
    batch = 25
    momentum_param = 0.25
    epochs=20
    # np.random.seed(0)
    results = mlp_batch(data=loaded_data, alpha=alpha, draw_range=0.2, batch_size=batch, epochs=epochs,
                        images_len_divider=1, momentum_param=momentum_param)
    save(data=results, filename=f'test_weights_22_alpha_{alpha}_batch_{batch}_'
                                f'momentum_{momentum_param}_epochs_{epochs}.pkl')