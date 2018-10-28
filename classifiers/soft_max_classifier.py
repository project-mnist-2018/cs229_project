""" SoftMax MNIST classifier """
from classifiers.utils.misc import get_real_mnist, plot_mist
from classifiers.utils.preprocessing import preprocess_raw_mnist_data
import tensorflow as tf
from tensorflow import keras


def simple_soft_max_classifier():
    """ This function returns a simple softmax keras classifier
    :return:    keras untrained simple soft_max classifier
    """
    classifier = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

    classifier.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return classifier


def main(plot=False):
    """ Main function """
    # Get mnist train and test dataset
    (x_train, y_train), (x_test, y_test) = get_real_mnist()

    # Preprocess raw data
    print('preprocess raw data')
    x_train = preprocess_raw_mnist_data(x_train)
    x_test = preprocess_raw_mnist_data(x_test)

    # Build classifier
    sm_clf = simple_soft_max_classifier()

    # Train classifier
    print('\ntrain the classifier')
    sm_clf.fit(x_train, y_train, epochs=5)

    print('\ntest the classifier')
    test_loss, test_acc = sm_clf.evaluate(x_test, y_test)

    print('\n#######################################')
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    if plot:
        plot_mist(x_train, y_train, 9, save_file_path='plots/test.png')


if __name__ == '__main__':
    main()
