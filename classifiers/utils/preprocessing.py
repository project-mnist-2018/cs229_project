""" Input preprocessing functions """


def preprocess_raw_mnist_data(x):
    """ Preprocessing the data """
    # Reshape x
    x = x.astype('float32')

    # Transform all input matrix elements in values belonging to [0,1] interval
    x /= 255

    return x