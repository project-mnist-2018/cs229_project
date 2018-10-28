""" Input preprocessing functions """


def preprocess_raw_mnist_data(x):
    """ Preprocessing the data """
    # Reshape x
    x = x.astype('float32')

    # Transform all input values to 0 to 1 values
    x /= 255

    return x