from loader.loader import load
from loader.mnist_loader import load_data_wrapper
from network import check_result

def print_result(filename):
    weights = load(filename=filename)
    _, _, test_data = load_data_wrapper("../data")
    te_in, te_out = test_data
    for w in weights:
        print(check_result(*w, te_in, te_out))


if __name__ == "__main__":
    print_result('test_weights_2.pkl')