""" Convolutional Neural Network MNIST classifier """
from utils.misc import get_real_mnist, get_gan_mnist, plot_mist
from utils.preprocessing import preprocess_raw_mnist_data
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations
import numpy as np

def cnn_classifier():
    """ This function returns a Convolutional NN keras classifier
    :return:    keras untrained Convolutional NN soft_max classifier
    """
    classifier = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=5,
                            strides=1,
                            activation='relu',
                            input_shape=(28, 28, 1)),
        keras.layers.Conv2D(64, kernel_size=5,
                            strides=1,
                            activation='relu',
                            input_shape=(24, 24, 1)),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
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
    x_train = preprocess_raw_mnist_data(x_train, conv=True)
    x_test = preprocess_raw_mnist_data(x_test, conv=True)
    x_gan_test = preprocess_raw_mnist_data(x_gan_test, conv=True)


    # Build classifier
    cnn_clf = cnn_classifier()

    epochs = 5

    if train:
        # Train classifier
        print('\ntrain the classifier')

        #history = cnn_clf.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        history = cnn_clf.fit(x_train, y_train, epochs=epochs, validation_split=0.1)

        # Save weights
        cnn_clf.save_weights('weights/cnn_clf_%s.h5' % epochs)

        #Plots train and validation datasets
        #Get data from history
        print(history.history.keys())
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title("model accuracy")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("output/fully_connected_model_accuracy.png")
        plt.show()
        #Save the plot

        #Plot the loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("output/fully_connected_model_loss.png")
        plt.show()

    else:
        # Load the model weights
        import os
        weights_file_path = os.path.abspath(os.path.join(os.curdir, 'weights/cnn_clf_%s.h5' % epochs))
        if not print(os.path.exists(weights_file_path)):
            print("The weights file path specified does not exists: %s" % os.path.exists(weights_file_path))
        cnn_clf.load_weights(weights_file_path)

    print('\ntest the classifier')
    test_loss, test_acc = cnn_clf.evaluate(x_test, y_test)

    print('\n#######################################')
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    print('\ntest the classifier on gan mnist')
    test_loss, test_acc = cnn_clf.evaluate(x_gan_test[:100], y_gan_test[:100])

    print('\n#######################################')
    print('Test loss gan:', test_loss)
    print('Test accuracy gan:', test_acc)


    class_idx = 0
    indices = np.where(y_test[:, class_idx] == 1.)[0]

    # pick some random input from here.
    idx = indices[0]

    # Lets sanity check the picked image.
    plt.rcParams['figure.figsize'] = (18, 6)

    plt.imshow(x_test[idx][..., 0])

    if plot:
        plot_mist(x_train, y_train, 9, save_file_path='plots/test.png')


if __name__ == '__main__':
    main(train=True)
