import numpy as np
import matplotlib.pyplot as plt

from loader.loader import load


def set_plt_data(plt, title, names, xlabel, ylabel='test data accuracy'):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(names, loc='lower right')
    # # TODO
    # locs, _ = plt.xticks()
    # plt.xticks([5] + locs)
    # print(locs)
    plt.show()


def calc_avg_duration(input):
    max_len = max(len(sub_list) for sub_list in input)
    differencies_acc = [0] * max_len
    for sub_list in input:
        sub_list = list(map(lambda x: float(x), sub_list))
        differencies = [sub_list[0]] + [sub_list[i] - sub_list[i - 1] for i in range(1, len(sub_list))]
        alligned_differencies = differencies + [differencies[-1]] * (max_len - len(differencies))
        differencies_acc = np.asarray(alligned_differencies) + np.asarray(differencies_acc)
    avg_differencies = differencies_acc / len(input)
    return list(np.cumsum(avg_differencies))


def prepare_avg_param(filename, param_name, ommit_first=True):
    data = load(filename)
    max_epoch = max(d['epochs'] for d in data['many_results']) + 1
    many_accuracies = [d['test_accuracies'] for d in data['many_results']]
    alligned_many_accuracies = [elem + [elem[-1]] * (max_epoch - len(elem)) for elem in many_accuracies]
    avg_accuracies = np.asarray(alligned_many_accuracies).sum(axis=0) / len(alligned_many_accuracies)
    if ommit_first:
        # TODO
        avg_accuracies = avg_accuracies[1:].round(4)
        max_epoch -= 1
    elapsed_times = [d['elapsed_times'] if 'elapsed_times' in d else None for d in data['many_results']]
    avg_times = calc_avg_duration(elapsed_times) if all(elapsed_times) else None
    return {param_name: data[param_name], 'avg_accuracies': avg_accuracies, 'max_epoch': max_epoch,
            'activation': data['activation'] if 'activation' in data else '', 'avg_times': avg_times,
            'use_adagrad': data['use_adagrad'] if 'use_adagrad' in data else False,
            'draw_type': data['draw_type'] if 'draw_type' in data else 'uniform',
            'momentum_param': data['momentum_param']}


def plot_param_to_epochs_for_many_results(multiple_data, param_name, print_draw_type=False, is_log=False):
    names = []
    for data_dict in multiple_data:
        names.append(data_dict['activation'] + ', ')
        if data_dict['use_adagrad']:
            names[-1] += 'adagrad, '
        if print_draw_type:
            names[-1] += f'''{data_dict['draw_type']}, '''
        if param_name == 'draw_range':
            names[-1] += str(f'-{data_dict[param_name]}:{data_dict[param_name]}')
        else:
            names[-1] += str(data_dict[param_name])
        # TODO
        plt.plot(range(0, data_dict['max_epoch']), data_dict['avg_accuracies'])
    if is_log:
        plt.yscale('log')
    title=f'MLP\n{param_name}'
    set_plt_data(plt, title, names, xlabel='epochs')


def plot_param_to_time_for_many_results(multiple_data, param_name):
    names = []
    for data_dict in multiple_data:
        names.append(f'{data_dict["activation"]}, ')
        if param_name == 'draw_range':
            names[-1] += str(f'-{data_dict[param_name]}:{data_dict[param_name]}')
        else:
            names[-1] += str(data_dict[param_name])
        plt.plot(data_dict['avg_times'], data_dict['avg_accuracies'])
    title=f'MLP\n{param_name}'
    set_plt_data(plt, title, names, xlabel='time [sec]')


def plot_epoch_to_time_for_many_results(multiple_data, param_name):
    names = []
    for data_dict in multiple_data:
        names.append(data_dict["activation"] + ', ')
        if data_dict['use_adagrad']:
            names[-1] += 'adagrad, '
        names[-1] += str(data_dict[param_name])
        plt.plot(data_dict['avg_times'], range(data_dict['max_epoch']))
    title=f'MLP\n{param_name}'
    set_plt_data(plt, title, names, xlabel='time [sec]', ylabel='epochs')


def draw_chart(filenames, param_name, print_draw_type=False):
    multiple_data = [prepare_avg_param(filename=f, param_name=param_name) for f in filenames]
    print_results(multiple_data, param_name=param_name)
    plot_param_to_epochs_for_many_results(multiple_data, param_name=param_name, print_draw_type=print_draw_type)
    if all(d['avg_times'] for d in multiple_data):
        plot_param_to_time_for_many_results(multiple_data, param_name=param_name)
        plot_epoch_to_time_for_many_results(multiple_data, param_name=param_name)


def print_results(multiple_data, param_name):
    max_size = max(data['avg_accuracies'].shape[0] for data in multiple_data)
    print(param_name)
    print([data[param_name] for data in multiple_data])
    to_print = np.ones(shape=(max_size, len(multiple_data))) * -1
    for i in range(len(multiple_data)):
        data = multiple_data[i]['avg_accuracies'].round(4)
        to_print[0:len(data), i] = data
    print(to_print)


def basic_simulations():
    draw_chart(
        param_name='draw_range',
        filenames=[
            'draw_range_simulation_sigmoid_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_24.4_times_5.pkl',
            'draw_range_simulation_sigmoid_alpha_0.04_batch_100_draw_range_0.4_hidden_neurones_50_avg_epochs_24.2_times_5.pkl',
            'draw_range_simulation_sigmoid_alpha_0.04_batch_100_draw_range_0.6_hidden_neurones_50_avg_epochs_25.0_times_5.pkl',
            'draw_range_simulation_sigmoid_alpha_0.04_batch_100_draw_range_0.8_hidden_neurones_50_avg_epochs_23.8_times_5.pkl',
            'draw_range_simulation_sigmoid_alpha_0.04_batch_100_draw_range_1.0_hidden_neurones_50_avg_epochs_27.6_times_5.pkl',
        ],
    )
    draw_chart(
        param_name='alpha',
        filenames=[
            'alpha_simulation_sigmoid_alpha_0.005_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_51.4_times_5.pkl',
            'alpha_simulation_sigmoid_alpha_0.01_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_54.0_times_5.pkl',
            'alpha_simulation_sigmoid_alpha_0.02_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_31.8_times_5.pkl',
            'alpha_simulation_sigmoid_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_24.4_times_5.pkl',
            'alpha_simulation_sigmoid_alpha_0.08_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_19.6_times_5.pkl',
        ],
    )
    draw_chart(
        param_name='batch_size',
        filenames=[
            'batch_simulation_sigmoid_alpha_0.04_batch_5_draw_range_0.2_hidden_neurones_50_avg_epochs_22.8_times_5.pkl',
            'batch_simulation_sigmoid_alpha_0.04_batch_10_draw_range_0.2_hidden_neurones_50_avg_epochs_26.2_times_5.pkl',
            'batch_simulation_sigmoid_alpha_0.04_batch_25_draw_range_0.2_hidden_neurones_50_avg_epochs_23.2_times_5.pkl',
            'batch_simulation_sigmoid_alpha_0.04_batch_50_draw_range_0.2_hidden_neurones_50_avg_epochs_21.4_times_5.pkl',
            'batch_simulation_sigmoid_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_24.0_times_5.pkl',
            'batch_simulation_sigmoid_alpha_0.04_batch_200_draw_range_0.2_hidden_neurones_50_avg_epochs_31.4_times_5.pkl',
        ],
    )
    draw_chart(
        param_name='hidden_neurones',
        filenames=[
            'hidden_neurones_simulation_sigmoid_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_15_avg_epochs_17.8_times_5.pkl',
            'hidden_neurones_simulation_sigmoid_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_25_avg_epochs_24.6_times_5.pkl',
            'hidden_neurones_simulation_sigmoid_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_24.0_times_5.pkl',
            'hidden_neurones_simulation_sigmoid_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_75_avg_epochs_26.2_times_5.pkl',
            'hidden_neurones_simulation_sigmoid_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_100_avg_epochs_26.6_times_5.pkl',
        ],
    )

    # sigmoid vs relu
    draw_chart(
        param_name='batch_size',
        filenames=[
            'batch_simulation_sigmoid_alpha_0.04_batch_5_draw_range_0.2_hidden_neurones_50_avg_epochs_22.8_times_5.pkl',
            'batch_simulation_sigmoid_alpha_0.04_batch_10_draw_range_0.2_hidden_neurones_50_avg_epochs_26.2_times_5.pkl',
            'batch_simulation_sigmoid_alpha_0.04_batch_25_draw_range_0.2_hidden_neurones_50_avg_epochs_23.2_times_5.pkl',
            'batch_simulation_sigmoid_alpha_0.04_batch_50_draw_range_0.2_hidden_neurones_50_avg_epochs_21.4_times_5.pkl',

            'batch2_simul_relu_alpha_0.01_batch_5_draw_r_0.01_hidden_n_50_mom_0_avg_epochs_12.0_reps_2.pkl',
            'batch2_simul_relu_alpha_0.01_batch_10_draw_r_0.01_hidden_n_50_mom_0_avg_epochs_11.5_reps_2.pkl',
            'batch2_simul_relu_alpha_0.01_batch_25_draw_r_0.01_hidden_n_50_mom_0_avg_epochs_13.5_reps_2.pkl',
            'batch2_simul_relu_alpha_0.01_batch_50_draw_r_0.01_hidden_n_50_mom_0_avg_epochs_7.0_reps_2.pkl',
        ],
    )
    draw_chart(
        param_name='hidden_neurones',
        filenames=[
            'hidden_neurones_simulation_sigmoid_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_15_avg_epochs_17.8_times_5.pkl',
            'hidden_neurones_simulation_sigmoid_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_25_avg_epochs_24.6_times_5.pkl',
            'hidden_neurones_simulation_sigmoid_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_24.0_times_5.pkl',
            'hidden_neurones_simulation_sigmoid_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_75_avg_epochs_26.2_times_5.pkl',

            'hidden_neurones_simulation_relu_alpha_0.01_batch_5_draw_range_0.2_hidden_neurones_15_avg_epochs_12.8_times_5.pkl',
            'hidden_neurones_simulation_relu_alpha_0.01_batch_5_draw_range_0.2_hidden_neurones_25_avg_epochs_13.2_times_5.pkl',
            'hidden_neurones_simulation_relu_alpha_0.01_batch_5_draw_range_0.2_hidden_neurones_50_avg_epochs_10.4_times_5.pkl',
            'hidden_neurones_simulation_relu_alpha_0.01_batch_5_draw_range_0.2_hidden_neurones_75_avg_epochs_10.8_times_5.pkl',
        ],
    )


def advanced_simulations():
    draw_chart(
        param_name='draw_range',
        print_draw_type=True,
        filenames=[
            'draw_range___simul_sigmoid_alpha_0.04_batch_100_draw_r_0.2_hidden_n_50_mom_0_avg_epochs_35.0_reps_4.pkl',
            'draw_range___simul_sigmoid_alpha_0.04_batch_100_draw_r_0.6_hidden_n_50_mom_0_avg_epochs_33.0_reps_4.pkl',
            'draw_range___simul_sigmoid_alpha_0.04_batch_100_draw_r_1.0_hidden_n_50_mom_0_avg_epochs_35.5_reps_4.pkl',
            'he_normal___simul_sigmoid_alpha_0.04_batch_100_draw_r_0.2_hidden_n_50_mom_0_avg_epochs_33.75_reps_4.pkl',
            'he_normal___simul_sigmoid_alpha_0.04_batch_100_draw_r_0.6_hidden_n_50_mom_0_avg_epochs_34.75_reps_4.pkl',
            'he_normal___simul_sigmoid_alpha_0.04_batch_100_draw_r_1.0_hidden_n_50_mom_0_avg_epochs_35.0_reps_4.pkl',
            'xavier___simul_sigmoid_alpha_0.04_batch_100_draw_r_0.2_hidden_n_50_mom_0_avg_epochs_38.25_reps_4.pkl',
            'xavier___simul_sigmoid_alpha_0.04_batch_100_draw_r_0.6_hidden_n_50_mom_0_avg_epochs_29.0_reps_4.pkl',
            'xavier___simul_sigmoid_alpha_0.04_batch_100_draw_r_1.0_hidden_n_50_mom_0_avg_epochs_29.75_reps_4.pkl',
        ],
    )

    # adagrad
    draw_chart(
        param_name='alpha',
        filenames=[
            'alpha_adagrad___simul_sigmoid_alpha_0.01_batch_100_draw_r_0.2_hidden_n_50_mom_0_avg_epochs_112.5_reps_4.pkl',
            'alpha_adagrad___simul_sigmoid_alpha_0.02_batch_100_draw_r_0.2_hidden_n_50_mom_0_avg_epochs_58.0_reps_4.pkl',
            'alpha_adagrad___simul_sigmoid_alpha_0.04_batch_100_draw_r_0.2_hidden_n_50_mom_0_avg_epochs_56.0_reps_4.pkl',
            'alpha_adagrad___simul_sigmoid_alpha_0.1_batch_100_draw_r_0.2_hidden_n_50_mom_0_avg_epochs_54.0_reps_4.pkl',
            'alpha___simul_sigmoid_alpha_0.01_batch_100_draw_r_0.2_hidden_n_50_mom_0_avg_epochs_78.0_reps_4.pkl',
            'alpha___simul_sigmoid_alpha_0.02_batch_100_draw_r_0.2_hidden_n_50_mom_0_avg_epochs_54.5_reps_4.pkl',
            'alpha___simul_sigmoid_alpha_0.04_batch_100_draw_r_0.2_hidden_n_50_mom_0_avg_epochs_32.25_reps_4.pkl',
        ],
    )

    draw_chart(
        param_name = 'momentum_param',
        filenames=[
            'momentum___simul_sigmoid_alpha_0.03_batch_100_draw_r_0.2_hidden_n_50_mom_0_avg_epochs_34.5_reps_4.pkl',
            'momentum___simul_sigmoid_alpha_0.03_batch_100_draw_r_0.2_hidden_n_50_mom_0.25_avg_epochs_34.0_reps_4.pkl',
            'momentum___simul_sigmoid_alpha_0.03_batch_100_draw_r_0.2_hidden_n_50_mom_0.5_avg_epochs_34.25_reps_4.pkl',
            'momentum___simul_sigmoid_alpha_0.03_batch_100_draw_r_0.2_hidden_n_50_mom_0.75_avg_epochs_29.0_reps_4.pkl',
        ],
    )

    draw_chart(
        param_name='alpha',
        print_draw_type=True,
        filenames=[
            'alpha___simul_sigmoid_alpha_0.04_batch_100_draw_r_0.2_hidden_n_50_mom_0_avg_epochs_32.25_reps_4.pkl',
            'alpha_adagrad___simul_sigmoid_alpha_0.1_batch_100_draw_r_0.2_hidden_n_50_mom_0_avg_epochs_54.0_reps_4.pkl',
            'momentum___simul_sigmoid_alpha_0.03_batch_100_draw_r_0.2_hidden_n_50_mom_0.5_avg_epochs_34.25_reps_4.pkl',
            'he_normal___simul_sigmoid_alpha_0.04_batch_100_draw_r_0.2_hidden_n_50_mom_0_avg_epochs_33.75_reps_4.pkl',
            'xavier___simul_sigmoid_alpha_0.04_batch_100_draw_r_0.2_hidden_n_50_mom_0_avg_epochs_38.25_reps_4.pkl',
        ],
    )


if __name__ == "__main__":
    basic_simulations()
    advanced_simulations()
