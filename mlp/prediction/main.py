from loader.loader import load
from loader.mnist_loader import load_data_wrapper
from prediction.network import calc_prediction_accuracy
from utils.timer import elapsed_timer

def print_result(filename, test_data):
    print(filename)
    with elapsed_timer() as timer:
        te_in, te_out = test_data
        weights = load(filename=filename)['weights']
        for i in range(len(weights)):
            print(f'{i}   {calc_prediction_accuracy(*weights[i], te_in, te_out)}')
            print(f'timer: {timer():.2f}')


if __name__ == "__main__":
    _, validation_data , test_data = load_data_wrapper("../data")
    # print_result('test_weights.pkl', test_data)
    # print_result('test_weights_12_bias_1.pkl', test_data)
    # print_result('test_weights_12_bias_1.pkl', validation_data)

    # print_result('test_weights_12_bias_1.pkl', test_data)
    # print_result('test_weights_18_bias_1_batch_1.pkl', test_data)

    # print_result('test_weights_20_bias_1_batch_1_momentum_0.3.pkl', test_data)
    # print_result('test_weights_20_bias_1_batch_1_momentum_0.3.pkl', validation_data)


    # print_result('test_weights_22_alpha_0.007_bias_1_batch_25_momentum_0.25_res_0.9644.pkl', test_data)
    # print_result('test_weights_22_alpha_0.007_bias_1_batch_25_momentum_0.25_res_0.9644.pkl', validation_data)

    # filename='validation_alpha_0.03_batch_100_draw_range_0.2_epochs_24_res_0.9688_momentum_param_0.3.pkl'
    # print_result(filename=filename, test_data=test_data)
    filename = 'validation_alpha_0.03_batch_100_draw_range_0.2_epochs_18_res_0.966_momentum_param_0.3.pkl'
    print_result(filename=filename, test_data=test_data)
