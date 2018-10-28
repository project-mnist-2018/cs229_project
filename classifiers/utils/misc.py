""" Miscellaneous auxiliary functions """
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy


# TODO mock function that creates a dummy feature vector
def get_feature_vector():
    return numpy.zeros(28*28*256)


def get_real_mnist():
    # load mnist dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return (X_train, y_train), (X_test, y_test)


# TODO create the function for importing the gan generated data manually labeled
#      from data folder
def get_gan_mnist():
    pass


def plot_mist(x, y, num_img, save_file_path=None):
    """ Auxiliary function that plots mnist images

    :param x:                   minst images
    :param y:                   mnist labes
    :param num_img:             number of mnist images to plot
    :param save_file_path:      (optional) path where to save the plot
    :return:                    None
    """
    size = numpy.ceil(numpy.sqrt(num_img))
    for i in range(num_img):
        plt.subplot(size, size, i + 1)
        plt.tight_layout()
        plt.imshow(x[i], cmap='gray', interpolation='none')
        plt.title(y[i])
        plt.xticks([])
        plt.yticks([])

    if save_file_path:
        plt.savefig(save_file_path)

    plt.show()