""" Random MNIST classifier """
from random import randint
from classifiers.utils.misc import get_feature_vector
import numpy


def random_classifier(x):
    """
    :param x:   feature vector
    :return:    prediction, a zero vector with only one element set to 1,
                which index number represent the predicted number.
                E.g: array([ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.])
                represents prediction: 3
    """
    output = numpy.zeros(10)
    output[randint(0,9)] = 1.0

    return output


def main():
    """ Main function """
    x = get_feature_vector()
    prediction = random_classifier(x)
    print(prediction)


if __name__ == "__main__":
    main()