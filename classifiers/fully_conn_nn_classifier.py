""" Fully connected Neural Network MNIST classifier """
from utils.misc import get_real_mnist, get_gan_mnist, plot_mist
from utils.preprocessing import preprocess_raw_mnist_data
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def fcnn_classifier():
    """ This function returns a Fully Connected NN keras classifier
    :return:    keras untrained Fully Connected NN soft_max classifier
    """
    classifier = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    classifier.compile(optimizer=tf.train.AdamOptimizer(),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    return classifier


def main(plot=False, train=True):
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
    fcnn_clf = fcnn_classifier()

    epochs = 100

    if train:
        # Train classifier
        print('\ntrain the classifier')

        #history = fcnn_clf.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        history = fcnn_clf.fit(x_train, y_train, epochs=epochs, validation_split=0.1)

        # Save weights
        fcnn_clf.save_weights('weights/fcnn_clf_%s.h5' % epochs)


        #Get data from history
        print(history.history.keys())
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title("model accuracy")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("output/fully_connected_model_accuracy.png")
        plt.show()
        #Save the plot

        #Plot the loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("output/fully_connected_model_loss.png")
        plt.show()
    else:
        # Load the model weights
        import os
        weights_file_path = os.path.abspath(os.path.join(os.curdir, 'weights/fcnn_clf_%s.h5' % epochs))
        if not print(os.path.exists(weights_file_path)):
            print("The weights file path specified does not exists: %s" % os.path.exists(weights_file_path))
        fcnn_clf.load_weights(weights_file_path)

    print('\ntest the classifier')
    test_loss, test_acc = fcnn_clf.evaluate(x_test[:1000], y_test[:1000])

    print('\n#######################################')
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    print('\ntest the classifier on gan mnist')
    test_loss, test_acc = fcnn_clf.evaluate(x_gan_test[:1000], y_gan_test[:1000])

    print('\n#######################################')
    print('Test loss gan:', test_loss)
    print('Test accuracy gan:', test_acc)

    if plot:
        plot_mist(x_train, y_train, 9, save_file_path='plots/test.png')


if __name__ == '__main__':
    main()

