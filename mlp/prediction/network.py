import numpy as np

from utils.timer import elapsed_timer

HIDDEN_BIAS = 1


def calc_prediction_accuracy(activation_func, hidden_weight, output_weight, test_input, test_output):
    net_hidden = test_input @ hidden_weight
    hidden_with_bias = np.ones(shape=(net_hidden.shape[0], net_hidden.shape[1] + HIDDEN_BIAS))
    hidden_with_bias[:, HIDDEN_BIAS:] = activation_func(net_hidden)
    net_output = hidden_with_bias @ output_weight
    output = activation_func(net_output)
    res = output
    res = np.argmax(res, axis=1)
    output_res = np.argmax(test_output, axis=1)
    return np.sum(output_res == res) / output_res.shape[0]


def draw_weights(draw_range, input_len, output_len):
    return np.random.uniform(low=-draw_range, high=draw_range, size=(input_len * output_len)) \
        .reshape((input_len, output_len))


def mlp(data, activation, alpha, draw_range, batch_size, hidden_neurones, worse_result_limit=2, momentum_param=0,
        is_adagrad=False, images_len_divider=1):
    training_data, validation_data, test_data = data
    tr_in, tr_out = (data[0: len(data) // images_len_divider] for data in training_data)
    activation_func, activation_func_prim = activation.activation, activation.activation_prim

    images_len, input_data_len = tr_in.shape[0], tr_in.shape[1]
    hidden_neurones_size, output_neurones_size = hidden_neurones, tr_out.shape[1]
    res_weights, validation_accuracies, test_accuracies, elapsed_times = [], [], [], []

    weights_hidden = draw_weights(draw_range=draw_range, input_len=input_data_len, output_len=hidden_neurones_size)
    hidden_neurones_size = hidden_neurones_size + HIDDEN_BIAS
    weights_output = draw_weights(draw_range=draw_range, input_len=hidden_neurones_size,
                                  output_len=output_neurones_size)

    res_weights.append([weights_hidden, weights_output])
    validation_accuracies.append(calc_prediction_accuracy(activation_func, weights_hidden, weights_output, *validation_data))
    print('validation_accuracies: ', validation_accuracies[-1])
    test_accuracies.append(calc_prediction_accuracy(activation_func, weights_hidden, weights_output, *test_data))
    print('result: ', test_accuracies[-1])

    epochs, worse_result_counter = 0, 0
    assert (images_len % batch_size == 0)
    split_len = images_len / batch_size
    with elapsed_timer() as timer:
        while worse_result_counter < worse_result_limit:
            weights1_delta_prev = 0
            weights2_delta_prev = 0
            for tr_in_batched, tr_out_batched in zip(np.split(tr_in, split_len), np.split(tr_out, split_len)):
                net_hidden = tr_in_batched @ weights_hidden
                hidden = activation_func(net_hidden)
                hidden_with_bias = np.ones(shape=(hidden.shape[0], hidden.shape[1] + HIDDEN_BIAS))
                hidden_with_bias[:, HIDDEN_BIAS:] = hidden

                net_output = hidden_with_bias @ weights_output
                output = activation_func(net_output)

                err_out = (tr_out_batched - output) * activation_func_prim(net_output)
                err_hidden = err_out @ weights_output[HIDDEN_BIAS:, :].transpose() * activation_func_prim(net_hidden)

                current_alpha = alpha
                if is_adagrad:
                    current_alpha = alpha * 1e-8


                weights2_delta = np.transpose(np.tile(hidden_with_bias, reps=(err_out.shape[1], 1, 1))) * err_out
                weights2_delta = np.sum(weights2_delta, axis=1) * current_alpha
                weights_output = weights_output + weights2_delta + weights2_delta_prev  * momentum_param
                weights2_delta_prev = weights2_delta

                weights1_delta = np.transpose(np.tile(tr_in_batched, reps=(err_hidden.shape[1], 1, 1))) * err_hidden
                weights1_delta = np.sum(weights1_delta, axis=1) * current_alpha
                weights_hidden = weights_hidden + weights1_delta + weights1_delta_prev * momentum_param
                weights1_delta_prev = weights1_delta

            epochs += 1
            res_weights.append([weights_hidden, weights_output])
            validation_accuracies.append(calc_prediction_accuracy(activation_func, weights_hidden, weights_output, *validation_data))
            if validation_accuracies[-1] > validation_accuracies[-2 - worse_result_counter]:
                worse_result_counter = 0
            else:
                worse_result_counter +=1
            test_accuracies.append(calc_prediction_accuracy(activation_func, weights_hidden, weights_output, *test_data))
            print(epochs, ' result: ', test_accuracies[-1])
            print(epochs, ' validation_accuracies', validation_accuracies[-1])
            print(epochs, f'timer: {timer():.2f}')
            elapsed_times.append(f'{timer():.2f}')
    print('final validation_accuracies', validation_accuracies)
    print(epochs, f'timer: {timer():.2f}')
    return {'weights': res_weights, 'test_accuracies': test_accuracies, 'epochs': epochs, 'elapsed_times': elapsed_times}
