""" SoftMax MNIST classifier """
from utils.misc import get_real_mnist, get_gan_mnist, plot_mist
from utils.preprocessing import preprocess_raw_mnist_data
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


def main(plot=False, train=False):
    """ Main function """
    # Get mnist train and test dataset
    (x_train, y_train), (x_test, y_test) = get_real_mnist()

    # Get gan test dataset
    (x_gan_test, y_gan_test) = get_gan_mnist()

    # Preprocess raw data
    print('preprocess raw data')
    x_train = preprocess_raw_mnist_data(x_train)
    x_test = preprocess_raw_mnist_data(x_test)
    x_gan_test = preprocess_raw_mnist_data(x_gan_test)

    # Build classifier
    sm_clf = simple_soft_max_classifier()

    epochs = 5

    if train:
        # Train classifier
        print('\ntrain the classifier')
        sm_clf.fit(x_train, y_train, epochs=epochs)

        # Save weights
        sm_clf.save_weights('weights/sm_clf_%s.h5' % epochs)

    else:
        # Load the model weights
        import os
        weights_file_path = os.path.abspath(os.path.join(os.curdir, 'weights/sm_clf_%s.h5' % epochs))
        if not print(os.path.exists(weights_file_path)):
            print("The weights file path specified does not exists: %s" % os.path.exists(weights_file_path))
        sm_clf.load_weights(weights_file_path)

    print('\ntest the classifier on real mnist')
    test_loss, test_acc = sm_clf.evaluate(x_test, y_test)

    print('\n#######################################')
    print('Test loss real:', test_loss)
    print('Test accuracy real:', test_acc)

    print('\ntest the classifier on gan mnist')
    test_loss, test_acc = sm_clf.evaluate(x_gan_test[:100], y_gan_test[:100])

    print('\n#######################################')
    print('Test loss gan:', test_loss)
    print('Test accuracy gan:', test_acc)

    if plot:
        # plot_mist(x_train, y_train, 9, save_file_path='plots/test.png')
        plot_mist(x_test, y_test, 36, save_file_path='plots/test.png')


if __name__ == '__main__':
    main(False)
