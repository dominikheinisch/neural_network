import matplotlib.pyplot as plt

from file_io import read


def set_plt_data(plt, title, names, x_name):
    plt.title(title)
    plt.ylabel('avg epochs')
    plt.xlabel(x_name)
    plt.legend(names, loc='upper right')
    plt.show()


def plot_param_to_epochs(files, is_log=False):
    names = []
    for f in files:
        loaded_dict = read(f)
        algo_name = loaded_dict['algo_name']
        optional_err_desc = f" for max_err={loaded_dict['max_err']}" if algo_name.find('ADALINE') == 0 else ''
        names.append(algo_name + optional_err_desc)
        x_name = loaded_dict['params_name']
        title = f"simulation {x_name}"
        plt.plot(loaded_dict['params'], loaded_dict['avg_epochs'])
    # x_list = loaded_dict['params']
    # x_stick = [[-x, x] for x in x_list]
    # plt.xticks(x_list, x_stick)
    plt.ylim(1, 100)
    if is_log:        plt.yscale('log')
    set_plt_data(plt, title, names, x_name)



if __name__ == "__main__":
    path_prefix = '../file_io/data/'
    filenames = [path_prefix + postfix for postfix in [
        'ADALINE_AND_alphas_0.0005_0.001_0.002_0.005_0.01_0.02_0.05_err_1.5.json',
        'ADALINE_AND_alphas_0.0005_0.001_0.002_0.005_0.01_0.02_0.05_err_2.0.json',
        'ADALINE_AND_alphas_0.0005_0.001_0.002_0.005_0.01_0.02_0.05_err_2.5.json',
        'PERCEPTRON_BINARY_AND_alphas_0.0005_0.001_0.002_0.005_0.01_0.02_0.05.json',
        'PERCEPTRON_BIPOLAR_AND_alphas_0.0005_0.001_0.002_0.005_0.01_0.02_0.05.json',
    ]]
    # plot_param_to_epochs(filenames)
    # plot_param_to_epochs(filenames, is_log=True)

    filenames = [path_prefix + postfix for postfix in [
        '__ADALINE_OR_alphas_0.0005_0.001_0.002_0.005_0.01_0.02_0.05_err_1.5.json',
        '__ADALINE_OR_alphas_0.0005_0.001_0.002_0.005_0.01_0.02_0.05_err_2.0.json',
        '__ADALINE_OR_alphas_0.0005_0.001_0.002_0.005_0.01_0.02_0.05_err_2.5.json',
        '__PERCEPTRON_BINARY_OR_alphas_0.0005_0.001_0.002_0.005_0.01_0.02_0.05.json',
        '__PERCEPTRON_BIPOLAR_OR_alphas_0.0005_0.001_0.002_0.005_0.01_0.02_0.05.json',
    ]]
    plot_param_to_epochs(filenames)
    plot_param_to_epochs(filenames, is_log=True)





    filenames = [path_prefix + postfix for postfix in [
        '_ADALINE_AND_weight_ranges_0.0_0.2_0.4_0.6_0.8_1_err_1.5.json',
        '_ADALINE_AND_weight_ranges_0.0_0.2_0.4_0.6_0.8_1_err_2.0.json',
        '_ADALINE_AND_weight_ranges_0.0_0.2_0.4_0.6_0.8_1_err_2.5.json',
        '_PERCEPTRON_BINARY_AND_weight_ranges_0.0_0.2_0.4_0.6_0.8_1.json',
        '_PERCEPTRON_BIPOLAR_AND_weight_ranges_0.0_0.2_0.4_0.6_0.8_1.json',
    ]]
    # plot_param_to_epochs(filenames)
    # plot_param_to_epochs(filenames, is_log=True)
