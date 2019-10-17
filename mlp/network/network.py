import numpy as np

from loader.mnist_loader import load_data, load_data_wrapper


if __name__ == "__main__":
    tr, va, te, = load_data("../data")
    tr_zip, va_zip, te_zip = load_data_wrapper("../data")
    tr_in, tr_out = tr_zip

    images_len = tr_in.shape[0]
    input_data_len = tr_in.shape[1]
    hidden_neurones_size = 50
    output_neurones_size = 10
    weights = np.random.uniform(-1, 1, neurones_size)
    weights = np.reshape(weights, (neurones_size, 1))

    print(weights.shape)
    print(tr_in.shape)

    output
    result = np.ones(shape=output.shape)
    for i in range(neurones_size):

        result[i] = input[i] @ weights
        result[i] = np.vectorize(result_func)(result[i])
        delta = output[i] - result[i]
        if(not delta == 0):
            is_finished = False
            weights = weights + delta * alpha * input[i]


    # a = 1
    # print('a')
    #
    # res = np.zeros((10, 1))
    # print(res)