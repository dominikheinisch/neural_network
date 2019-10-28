import matplotlib.pyplot as plt

from loader.loader import load


def set_plt_data(plt, title):
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('test data accuracy')
    plt.legend('mlp', loc='upper right')
    plt.show()


def plot_param_to_epochs(files, param_name, is_log=False):
    for f in files:
        loaded_dict = load(f)
        plt.plot(range(loaded_dict['epochs']), loaded_dict['accuracies'][1:])
    # x_list = loaded_dict['params']
    # x_stick = [[-x, x] for x in x_list]
    # plt.xticks(x_list, x_stick)
    # plt.ylim(1, 100)
    if is_log:        plt.yscale('log')
    title=f'mlp for {param_name}'
    set_plt_data(plt, title)



if __name__ == "__main__":
    filenames = [
        'validation_alpha_0.01_batch_100_draw_range_0.1_epochs_59_res_0.9645_momentum_param_0.pkl',
        'validation_alpha_0.03_batch_100_draw_range_0.1_epochs_29_res_0.9676_momentum_param_0.pkl',
        'validation_alpha_0.03_batch_100_draw_range_0.8_epochs_39_res_0.9645_momentum_param_0.3.pkl',
        'validation_alpha_0.03_batch_100_draw_range_1.0_epochs_45_res_0.9595_momentum_param_0.pkl',

    ]
    plot_param_to_epochs(filenames, param_name='draw_range')
    plot_param_to_epochs(filenames, param_name='draw_range', is_log=True)
