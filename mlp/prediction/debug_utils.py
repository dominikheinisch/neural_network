import numpy as np

from loader.loader import load
from loader.mnist_loader import load_data_wrapper
from saver.saver import save
# from prediction.network import calc_prediction_accuracy
from prediction.activation_function import SIGMOID, RELU
from utils.timer import elapsed_timer

# def print_result(filename, test_data):
#     activation_func = np.vectorize(SIGMOID[0])
#     print(filename)
#     with elapsed_timer() as timer:
#         te_in, te_out = test_data
#         weights = load(filename=filename)['weights']
#         for i in range(len(weights)):
#             print(f'{i}   {calc_prediction_accuracy(activation_func, *weights[i], te_in, te_out)}')
#             print(f'timer: {timer():.2f}')


def prepare(filename):
    data = load(filename)
    data['activation'] = 'sigmoid'
    str_to_find = '_simulation_'
    index = filename.find(str_to_find) + len(str_to_find)
    new_filename = f'{filename[:index]}sigmoid_{filename[index:]}'
    # data = load(filename)
    # hidden_neurones = 'hidden_neurones'
    # data[hidden_neurones] = 50
    # str_to_find = '_draw_range_'
    # index = filename.find(str_to_find) + len(str_to_find) + 4
    # new_filename = f'{filename[:index]}hidden_neurones_50_{filename[index:]}'
    # save(data=data, filename=new_filename)
    print("'" + new_filename + "',")


if __name__ == "__main__":
    _, validation_data , test_data = load_data_wrapper("../data")
    # print_result('test_weights.pkl', test_data)

    # filename = 'test_alpha_0.04_batch_100_draw_range_1.0_hidden_neurones_50.pkl'
    # print_result(filename=filename, test_data=test_data)

    # file = '2_test_alpha_0.01_batch_100_draw_range_0.05_hidden_neurones_25.pkl'
    # file = 'once_sigmoid_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_100_res_0.9739.pkl'
    # test_data = test_data[0], test_data[1]
    # print_result(filename=file, test_data=test_data)

    # from saver.saver import save
    files = [
        # 'draw_range_simulation_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_24.4_times_5.pkl',
        # 'draw_range_simulation_alpha_0.04_batch_100_draw_range_0.4_hidden_neurones_50_avg_epochs_24.2_times_5.pkl',
        # 'draw_range_simulation_alpha_0.04_batch_100_draw_range_0.6_hidden_neurones_50_avg_epochs_25.0_times_5.pkl',
        # 'draw_range_simulation_alpha_0.04_batch_100_draw_range_0.8_hidden_neurones_50_avg_epochs_23.8_times_5.pkl',
        # 'draw_range_simulation_alpha_0.04_batch_100_draw_range_1.0_hidden_neurones_50_avg_epochs_27.6_times_5.pkl',
        # 'alpha_simulation_alpha_0.005_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_51.4_times_5.pkl',
        # 'alpha_simulation_alpha_0.01_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_54.0_times_5.pkl',
        # 'alpha_simulation_alpha_0.02_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_31.8_times_5.pkl',
        # 'alpha_simulation_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_24.4_times_5.pkl',
        # 'alpha_simulation_alpha_0.08_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_19.6_times_5.pkl',
        # 'batch_simulation_alpha_0.04_batch_10_draw_range_0.2_hidden_neurones_50_avg_epochs_22.4_times_5.pkl',
        # 'batch_simulation_alpha_0.04_batch_25_draw_range_0.2_hidden_neurones_50_avg_epochs_25.8_times_5.pkl',
        # 'batch_simulation_alpha_0.04_batch_50_draw_range_0.2_hidden_neurones_50_avg_epochs_23.2_times_5.pkl',
        # 'batch_simulation_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_24.4_times_5.pkl',
        # 'batch_simulation_alpha_0.04_batch_200_draw_range_0.2_hidden_neurones_50_avg_epochs_24.2_times_5.pkl','
        # 'hidden_neurones_simulation_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_15_avg_epochs_22.6_times_5.pkl',
        # 'hidden_neurones_simulation_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_25_avg_epochs_20.6_times_5.pkl',
        # 'hidden_neurones_simulation_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_50_avg_epochs_24.4_times_5.pkl',
        # 'hidden_neurones_simulation_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_75_avg_epochs_26.0_times_5.pkl',
        # 'hidden_neurones_simulation_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_100_avg_epochs_31.2_times_5.pkl',
        # 'hidden_neurones_simulation_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_150_avg_epochs_38.0_times_2.pkl',
    ]
    #
    # # goodF = [
    # #     'hidden_neurones_simulation_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_15_avg_epochs_22.6_times_5.pkl',
    # #     'hidden_neurones_simulation_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_25_avg_epochs_20.6_times_5.pkl',
    # #     'hidden_neurones_simulation_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_75_avg_epochs_26.0_times_5.pkl',
    # #     'hidden_neurones_simulation_alpha_0.04_batch_100_draw_range_0.2_hidden_neurones_100_avg_epochs_31.2_times_5.pkl',
    # # ]
    # #
    #
    # for f in files:
    #     prepare(f)

    # print(load('once_relu_alpha_0.003_batch_10_draw_range_0.2_hidden_neurones_50.pkl'))

    print(load('hidden_neurones4_simulation_relu_alpha_0.01_batch_5_draw_range_0.2_hidden_neurones_25_avg_epochs_16.0_times_5.pkl'))

    # def calc_avg_times(input):
    #     max_len = max(len(sub_list) for sub_list in input)
    #     result_acc = [0] * max_len
    #     divider_acc = [0] * max_len
    #     for sub_list in input:
    #         for i in range(len(sub_list)):
    #             result_acc[i] += sub_list[i]
    #             divider_acc[i] += 1
    #     return [result_acc[i] / divider_acc[i] for i in range(max_len)]
    #
    # a = [[1, 3], [2, 4, 6, 9], [], [-1]]
    # print(calc_avg_times(a))
