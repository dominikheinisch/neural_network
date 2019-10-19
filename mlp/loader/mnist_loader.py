import pickle
import gzip
from os import path
import numpy as np

DATA_PATH =  'data'


def load_data(data_path=DATA_PATH):
    f = open(path.join(data_path, 'mnist.pkl'), 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper(data_path=DATA_PATH):
    tr_d, va_d, te_d = load_data(data_path)
    training_inputs, validation_inputs, test_inputs = [prepare_input(data) for data in [tr_d[0], va_d[0], te_d[0]]]
    training_results, validation_results, test_results = [vectorize_results(data)
                                                          for data in [tr_d[1], va_d[1], te_d[1]]]
    validation_data = (validation_inputs, validation_results)
    training_data = (training_inputs, training_results)
    test_data = (test_inputs, test_results)
    return training_data, validation_data, test_data


def prepare_input(input):
    res = np.zeros(shape=(input.shape[0], input.shape[1] + 1))
    res[:, 0] = 1
    res[:, 1:] = input
    return res


def vectorize_results(output):
    res = np.zeros(shape=(output.shape[0], 10))
    for i in range(output.shape[0]):
        res[i] = vectorize_result(output[i])
    return res


def vectorize_result(y):
    res = np.zeros(10)
    res[y] = 1.0
    return res


# def extract(data):
#     flattened_images = data[0]
#     result_numbers = data[1]
#     return [np.reshape(f, (28, 28)) for f in flattened_images], result_numbers


# def test_border(images):
#     def replace(img):
#         # img[2:26, 2:26] = np.zeros(shape=(24, 24))
#         return img
#     return np.sum(np.array([replace(img) for img in images]), axis=0)


# def border_cutter(images):
#     return [np.reshape(img, (28, 28)) for i in images]


if __name__ == "__main__":
    # print(2.66305923e-02 * 0.06037682)
    # print(1.60787056e-03)
    tr, va, te = load_data("../data")

    tr_zip, va_zip, te_zip = load_data_wrapper("../data")
    for inp, out in tr_zip:
        pass
        print(out)
