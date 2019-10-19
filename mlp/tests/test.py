from loader.loader import load
from loader.mnist_loader import load_data_wrapper
from network.network import check_result

def test_check_result():
    weights = load(filename='test_weights.pkl')
    _, _, test_data = load_data_wrapper("../data")
    te_in, te_out = test_data
    random_weights = weights[0]
    assert(check_result(*random_weights, te_in, te_out) == 0.0919)
    calculated_weights = weights[1]
    assert(check_result(*calculated_weights, te_in, te_out) == 0.5925)


if __name__ == "__main__":
    test_check_result()