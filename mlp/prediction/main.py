from loader.loader import load
from loader.mnist_loader import load_data_wrapper
from prediction.network import calc_prediction_accuracy


from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


def print_result(filename, test_data):
    te_in, te_out = test_data
    weights = load(filename=filename)
    print(filename)
    for i in range(len(weights)):
        print(f'{i}   {calc_prediction_accuracy(*weights[i], te_in, te_out)}')


if __name__ == "__main__":
    _, validation_data , test_data = load_data_wrapper("../data")
    # print_result('test_weights.pkl', test_data)
    # print_result('test_weights_12_bias_1.pkl', test_data)
    # print_result('test_weights_12_bias_1.pkl', validation_data)
    print_result('test_weights_12_bias_1.pkl', test_data)
    print_result('test_weights_18_bias_1_batch_1.pkl', test_data)
