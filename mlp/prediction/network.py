import numpy as np

from utils.timer import elapsed_timer

BIAS_SIZE = 1


def activate_with_bias(net, activation_func):
    activated_with_bias = np.ones(shape=(net.shape[0], net.shape[1] + BIAS_SIZE))
    activated_with_bias[:, BIAS_SIZE:] = activation_func(net)
    return activated_with_bias


def calc_prediction_accuracy(activation_func, hidden_weight, output_weight, test_input, test_output):
    net_hidden = test_input @ hidden_weight
    hidden_with_bias = activate_with_bias(net_hidden, activation_func)
    net_output = hidden_with_bias @ output_weight
    output = activation_func(net_output)
    res = np.argmax(output, axis=1)
    output_res = np.argmax(test_output, axis=1)
    return np.sum(output_res == res) / output_res.shape[0]


def add_bias(data):
    return np.append(np.ones(shape=(data[0].shape[0], BIAS_SIZE)), data[0], axis=1), data[1]


def draw_weights(draw_range, input_len, output_len):
    return np.random.uniform(low=-draw_range, high=draw_range, size=(input_len * output_len)) \
        .reshape((input_len, output_len))


def calc_delta(neurone_with_bias, error, current_alpha, delta_prev, momentum_param):
    temp = np.transpose(np.tile(neurone_with_bias, reps=(error.shape[1], 1, 1))) * error
    delta = np.sum(temp, axis=1) * current_alpha
    result = delta + delta_prev * momentum_param
    delta_prev[:] = delta
    return result


def mlp(data, activation, alpha, draw_range, batch_size, hidden_neurones, worse_result_limit=2, momentum_param=0,
        is_adagrad=False, images_len_divider=1):
    training_data, validation_data, test_data = [add_bias(d) for d in data]
    tr_in, tr_out = (data[: len(data) // images_len_divider] for data in training_data)
    activation_func, activation_func_prim = activation.activation, activation.activation_prim

    images_len, input_data_len = tr_in.shape[0], tr_in.shape[1]
    hidden_neurones_size, output_neurones_size = hidden_neurones, tr_out.shape[1]

    weights_hidden = draw_weights(draw_range=draw_range, input_len=input_data_len, output_len=hidden_neurones_size)
    hidden_neurones_size = hidden_neurones_size + BIAS_SIZE
    weights_output = draw_weights(draw_range=draw_range, input_len=hidden_neurones_size,
                                  output_len=output_neurones_size)

    res_weights, validation_accuracies, test_accuracies, elapsed_times = [], [], [], []
    epochs, worse_result_counter = 0, 0
    assert (images_len % batch_size == 0)
    split_len = images_len / batch_size
    with elapsed_timer() as timer:
        while worse_result_counter < worse_result_limit:
            res_weights.append([weights_hidden, weights_output])
            validation_accuracies.append(calc_prediction_accuracy(activation_func, weights_hidden, weights_output, *validation_data))
            if len(validation_accuracies) == 1:
                pass
            elif validation_accuracies[-1] > validation_accuracies[-2 - worse_result_counter]:
                worse_result_counter = 0
            else:
                worse_result_counter +=1
            test_accuracies.append(calc_prediction_accuracy(activation_func, weights_hidden, weights_output, *test_data))
            print(epochs, ' result: ', test_accuracies[-1])
            print(epochs, ' validation_accuracies', validation_accuracies[-1])

            if worse_result_counter < worse_result_limit:
                weights1_delta_prev, weights2_delta_prev = np.zeros(shape=(weights_hidden.shape)), np.zeros(shape=(weights_output.shape))
                for tr_in_batched, tr_out_batched in zip(np.split(tr_in, split_len), np.split(tr_out, split_len)):
                    net_hidden = tr_in_batched @ weights_hidden
                    hidden_with_bias = activate_with_bias(net_hidden, activation_func)

                    net_output = hidden_with_bias @ weights_output
                    output = activation_func(net_output)

                    err_out = (tr_out_batched - output) * activation_func_prim(net_output)
                    err_hidden = err_out @ weights_output[BIAS_SIZE:, :].transpose() * activation_func_prim(net_hidden)

                    current_alpha = alpha
                    if is_adagrad:
                        current_alpha = alpha * 1e-8

                    weights_output = weights_output + calc_delta(hidden_with_bias, err_out, current_alpha, weights2_delta_prev, momentum_param)
                    weights_hidden = weights_hidden + calc_delta(tr_in_batched, err_hidden, current_alpha, weights1_delta_prev, momentum_param)
                print(epochs, f'timer: {timer():.2f}')
                elapsed_times.append(f'{timer():.2f}')
                epochs += 1
    print('final validation_accuracies', validation_accuracies)
    print(epochs, f'timer: {timer():.2f}')
    return {'weights': res_weights, 'test_accuracies': test_accuracies, 'epochs': epochs, 'elapsed_times': elapsed_times}
