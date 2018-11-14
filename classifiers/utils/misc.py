""" Miscellaneous auxiliary functions """
import os

from keras.datasets import mnist
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy

# TODO mock function that creates a dummy feature vector
def get_feature_vector():
    return numpy.zeros(28*28)


def get_real_mnist():
    # load mnist dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return (X_train, y_train), (X_test, y_test)


def get_gan_mnist():
    gan_samples = []
    y = []
    for file_name in sorted(os.listdir('data/gan_samples'), key=lambda f: int(f.split('_')[-1].split('.')[0])):
        if 'npz' in file_name:
            npz = numpy.load('data/gan_samples/%s' % file_name)
            gan_samples.append(npz['x'])
        else:
            with open('data/gan_samples/%s' % file_name, 'r') as man_labels:
                y += man_labels.readlines()
    x = numpy.concatenate(gan_samples)

    # TODO manually label the gan immages in x
    y = y + ['?' for _ in range(len(x) - len(y))]

    return x, y


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
        plt.tight_layout(h_pad=-0.1, pad=-0.2)
        plt.imshow(x[i], cmap='gray', interpolation='none')
        plt.title(y[i])
        plt.xticks([])
        plt.yticks([])

    if save_file_path:
        if not os.path.exists('plots'):
            os.mkdir('plots')
        plt.savefig(save_file_path)

    plt.show()