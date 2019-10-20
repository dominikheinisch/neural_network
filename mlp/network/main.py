from loader.loader import load
from loader.mnist_loader import load_data_wrapper
from network import calc_prediction_accuracy

def print_result(filename, test_data):
    te_in, te_out = test_data
    weights = load(filename=filename)
    print(filename)
    for w in weights:
        print(calc_prediction_accuracy(*w, te_in, te_out))


if __name__ == "__main__":
    _, _, test_data = load_data_wrapper("../data")
    # print_result('test_weights.pkl', test_data)
    # print_result('test_weights_3.pkl', test_data)
    # print_result('test_weights_4.pkl', test_data)
    # print_result('test_weights_5.pkl', test_data)
    # print_result('test_weights_6.pkl', test_data)
    # print_result('test_weights_7_b0.pkl', test_data)
    print_result('test_weights_8_bias_1.pkl', test_data)
