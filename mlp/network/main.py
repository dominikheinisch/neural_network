from loader.loader import load
from loader.mnist_loader import load_data_wrapper
from network import calc_prediction_accuracy

def print_result(filename):
    weights = load(filename=filename)
    _, _, test_data = load_data_wrapper("../data")
    te_in, te_out = test_data
    for w in weights:
        print(calc_prediction_accuracy(*w, te_in, te_out))


if __name__ == "__main__":
    print_result('test_weights_2.pkl')