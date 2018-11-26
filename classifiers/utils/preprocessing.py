""" Input preprocessing functions """


def preprocess_raw_mnist_data(x, conv=False):
    """ Preprocessing the data """

    # Change element type
    x = x.astype('float32')

    # Transform all input matrix elements in values belonging to [0,1] interval
    x /= 255

    if conv:
        # Reshape if convolutional (channel last)
        x = x.reshape(x.shape[0], 28, 28, 1)

    return x
