from loader.loader import load
from loader.mnist_loader import load_data_wrapper
from prediction.network import calc_prediction_accuracy

def test_calc_prediction_accuracy():
    weights = load(filename='test_weights.pkl')
    _, _, test_data = load_data_wrapper("../data")
    te_in, te_out = test_data
    random_weights = weights[0]
    assert(calc_prediction_accuracy(*random_weights, te_in, te_out) == 0.0993)
    calculated_weights = weights[3]
    assert(calc_prediction_accuracy(*calculated_weights, te_in, te_out) == 0.7478)


# def test_hidden_backprop():



if __name__ == "__main__":
    test_calc_prediction_accuracy()
