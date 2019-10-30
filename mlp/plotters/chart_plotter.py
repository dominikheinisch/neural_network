import numpy as np
import matplotlib.pyplot as plt

from loader.loader import load


def set_plt_data(plt, title, names):
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('test data accuracy')
    plt.legend(names, loc='lower right')
    plt.show()


def plot_param_to_epochs(files, param_name, names, is_log=False):
    for f in files:
        loaded_dict = load(f)
        plt.plot(range(loaded_dict['epochs']), loaded_dict['accuracies'][1:])
    if is_log:
        plt.yscale('log')
    title=f'MLP for different {param_name}'
    set_plt_data(plt, title, names)


def plot_param_to_epochs_for_many_results(many_data, param_name, is_log=False):
    names = []
    for data_dict in many_data:
        if param_name == 'draw_range':
            names.append(f'-{data_dict[param_name]}:{data_dict[param_name]}')
        else:
            names.append(data_dict[param_name])
        plt.plot(range(data_dict['max_epoch']), data_dict['avg_accuracies'])
    if is_log:
        plt.yscale('log')
    title=f'mlp for {param_name}'
    set_plt_data(plt, title, names)


def prepare_avg_param(filename, param_name, ommit_first=True):
    data = load(filename)
    max_epoch = max(d['epochs'] for d in data['many_results']) + 1
    many_accuracies = [d['test_accuracies'] for d in data['many_results']]
    alligned_many_accuracies = [elem + [elem[-1]] * (max_epoch - len(elem)) for elem in many_accuracies]
    avg_accuracies = np.asarray(alligned_many_accuracies).sum(axis=0) / len(alligned_many_accuracies)
    if ommit_first:
        avg_accuracies = avg_accuracies[1:]
        max_epoch -= 1
    return {param_name: data[param_name], 'avg_accuracies': avg_accuracies, 'max_epoch': max_epoch}



def draw_chart(filenames, param_name):
    data = [prepare_avg_param(filename=f, param_name=param_name) for f in filenames]
    plot_param_to_epochs_for_many_results(data, param_name=param_name)

if __name__ == "__main__":
    draw_chart(
        param_name = 'draw_range',
        filenames=[
        'draw_range_simulation_alpha_0.04_batch_100_draw_range_0.2_avg_epochs_24.4_times_5.pkl',
        'draw_range_simulation_alpha_0.04_batch_100_draw_range_0.4_avg_epochs_24.2_times_5.pkl',
        'draw_range_simulation_alpha_0.04_batch_100_draw_range_0.6_avg_epochs_25.0_times_5.pkl',
        'draw_range_simulation_alpha_0.04_batch_100_draw_range_0.8_avg_epochs_23.8_times_5.pkl',
        'draw_range_simulation_alpha_0.04_batch_100_draw_range_1.0_avg_epochs_27.6_times_5.pkl',
        ],
    )

    draw_chart(
        param_name = 'alpha',
        filenames=[
        'alpha_simulation_alpha_0.005_batch_100_draw_range_0.2_avg_epochs_51.4_times_5.pkl',
        'alpha_simulation_alpha_0.01_batch_100_draw_range_0.2_avg_epochs_54.0_times_5.pkl',
        'alpha_simulation_alpha_0.04_batch_100_draw_range_0.2_avg_epochs_24.4_times_5.pkl',
        ],
    )

