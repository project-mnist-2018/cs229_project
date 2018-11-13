""" Fully connected Neural Network MNIST classifier """
from utils.misc import get_real_mnist, plot_mist
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

def main(plot=False):
    """ Main function """
    # Get mnist train and test dataset
    (x_train, y_train), (x_test, y_test) = get_real_mnist()

    # Preprocess raw data
    print('preprocess raw data')
    x_train = preprocess_raw_mnist_data(x_train)
    x_test = preprocess_raw_mnist_data(x_test)

    # Build classifier
    fcnn_clf = fcnn_classifier()

    # Train classifier
    print('\ntrain the classifier')
    history = fcnn_clf.fit(x_train, y_train, epochs=5)

    #Get data from history
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.title("model accuracy")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend('train', loc='upper left')
    plt.show()

    #Plot the loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend('train', loc='upper left')
    plt.show()

    print('\ntest the classifier')
    test_loss, test_acc = fcnn_clf.evaluate(x_test, y_test)

    print('\n#######################################')
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    if plot:
        plot_mist(x_train, y_train, 9, save_file_path='plots/test.png')


if __name__ == '__main__':
    main()

