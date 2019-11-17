import numpy as np

from prediction.activation_function import SIGMOID, RELU
from loader.mnist_loader import load_data_wrapper
from utils.timer import elapsed_timer

HIDDEN_BIAS = 1


def calc_prediction_accuracy(activation_func, hidden_weight, output_weight, test_input, test_output):
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


def mlp(data, activation, alpha, draw_range, batch_size, hidden_neurones, worse_result_limit=2, momentum_param=0,
        is_adagrad=False, images_len_divider=1):
    training_data, validation_data, test_data = data
    tr_in, tr_out = training_data
    activation_func = activation.activation
    activation_func_prim = activation.activation_prim

    images_len = tr_in.shape[0]
    assert(images_len % batch_size == 0)
    input_data_len = tr_in.shape[1]
    hidden_neurones_size = hidden_neurones
    output_neurones_size = 10
    weights = []
    validation_accuracies = []
    test_accuracies = []
    weights1 = np.random.uniform(low=-draw_range, high=draw_range, size=(input_data_len * hidden_neurones_size))
    weights1 = np.reshape(weights1, newshape=(input_data_len, hidden_neurones_size))

    hidden_neurones_with_bias_size = hidden_neurones_size + HIDDEN_BIAS
    weights2 = np.random.uniform(low=-draw_range, high=draw_range,
                                 size=(hidden_neurones_with_bias_size * output_neurones_size))
    weights2 = np.reshape(weights2, newshape=(hidden_neurones_with_bias_size, output_neurones_size))

    weights.append([weights1, weights2])
    validation_accuracies.append(calc_prediction_accuracy(activation_func, weights1, weights2, *validation_data))
    print('validation_accuracies: ', validation_accuracies[-1])
    test_accuracies.append(calc_prediction_accuracy(activation_func, weights1, weights2, *test_data))
    print('result: ', test_accuracies[-1])

    batch_indexes = [(i * batch_size, (i + 1) * batch_size) for i in range(images_len //
                                                                           images_len_divider // batch_size)]
    elapsed_times = []
    worse_result_counter = 0
    epochs = 0
    with elapsed_timer() as timer:
        while worse_result_counter < worse_result_limit:
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

                current_alpha = alpha
                if is_adagrad:
                    current_alpha = alpha * 1e-8


                weights2_delta = np.transpose(np.tile(hidden_with_bias,
                                                      reps=(err_out.shape[1], 1, 1))) * err_out
                weights2_delta = np.sum(weights2_delta, axis=1) * current_alpha
                weights2 = weights2 + weights2_delta + weights2_delta_prev  * momentum_param
                weights2_delta_prev = weights2_delta

                weights1_delta = np.transpose(np.tile(tr_in[batch_start:batch_end],
                                                      reps=(err_hidden.shape[1], 1, 1))) * err_hidden
                weights1_delta = np.sum(weights1_delta, axis=1) * current_alpha
                weights1 = weights1 + weights1_delta + weights1_delta_prev * momentum_param
                weights1_delta_prev = weights1_delta

            epochs += 1
            weights.append([weights1, weights2])
            validation_accuracies.append(calc_prediction_accuracy(activation_func, weights1, weights2, *validation_data))
            if validation_accuracies[-1] > validation_accuracies[-2 - worse_result_counter]:
                worse_result_counter = 0
            else:
                worse_result_counter +=1
            test_accuracies.append(calc_prediction_accuracy(activation_func, weights1, weights2, *test_data))
            print(epochs, ' result: ', test_accuracies[-1])
            print(epochs, ' validation_accuracies', validation_accuracies[-1])
            print(epochs, f'timer: {timer():.2f}')
            elapsed_times.append(f'{timer():.2f}')
    print('final validation_accuracies', validation_accuracies)
    print(epochs, f'timer: {timer():.2f}')
    return {'weights': weights, 'test_accuracies': test_accuracies, 'epochs': epochs, 'elapsed_times': elapsed_times}
