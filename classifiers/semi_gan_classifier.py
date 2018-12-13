""" Modified GAN semi-supervised learning MNIST classifier """
from utils.misc import get_real_mnist, get_gan_mnist, plot_mist
from utils.preprocessing import preprocess_raw_mnist_data
import tensorflow as tf
from tensorflow import keras
import numpy as np


def discriminator():
    model = keras.Sequential([
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
    ])

    # Define the model input
    model_input = keras.layers.Input(shape=(28, 28, 1))

    # obtain the sequential model output
    model_out = model(model_input)

    # Feed the sequential model output to both the discriminator and the classifier
    discr_out = keras.layers.Dense(1, activation="sigmoid")(model_out)
    clf_out = keras.layers.Dense(10 + 1, activation="softmax")(model_out)

    # Put together input, sequential, and output layers in one model
    model = keras.models.Model(inputs=model_input, outputs=[discr_out, clf_out])
    model.summary()

    # Compile the model
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
                  loss_weights=[0.5, 0.5],
                  metrics=['accuracy']
                  )

    return model


def main(plot=False, train=False):
    """ Main function """
    # Get mnist train and test dataset
    (x_train, y_train), (x_test, y_test) = get_real_mnist()

    # Get gan test dataset
    (x_gan_train, y_gan_train) = get_gan_mnist()

    # Preprocess raw data
    print('preprocess raw data')
    x_train = preprocess_raw_mnist_data(x_train, conv=True)
    x_test = preprocess_raw_mnist_data(x_test, conv=True)
    x_gan_train = preprocess_raw_mnist_data(x_gan_train, conv=True)


    # Build modified discriminator
    discrim = discriminator()


    epochs = 100

    # Combine real and synthetic data
    label_size = 5400
    unlabeled_size = 5400
    ratio = label_size / (label_size + unlabeled_size)

    if train:
        # Train classifier
        print('\ntrain the classifier')

        x_comb_train = np.append(x_gan_train[:unlabeled_size], x_train[:label_size], axis=0)
        y_comb_train = [
            np.append(np.zeros(unlabeled_size), np.ones(label_size), axis=0),
            # We assign the label 10 to all unlabeled examples as the digit that represents a fake image.
            np.append(np.zeros(unlabeled_size)+10, y_train[:label_size], axis=0)
        ]

        discrim.fit(x_comb_train, y_comb_train, epochs=epochs)

        # Save weights
        discrim.save_weights('weights/semi_sup_clf_%s_r%s.h5' % (epochs, ratio))

    else:
        # Load the model weights
        import os
        weights_file_path = os.path.abspath(os.path.join(os.curdir, 'weights/semi_sup_clf_%s_r%s.h5' % (epochs, ratio)))
        if not print(os.path.exists(weights_file_path)):
            print("The weights file path specified does not exists: %s" % os.path.exists(weights_file_path))
        discrim.load_weights(weights_file_path)

    print('\ntest the classifier')
    comb_test_loss, discr_test_loss, label_clf_test_loss, discr_test_acc, label_clf_test_acc  = discrim.evaluate(x_test[:1000], [np.ones(len(y_test[:1000])), y_test[:1000]])

    print('\n#######################################')
    print('Combined output layers Test loss:', comb_test_loss)
    print('Discriminator output layer Test loss:', discr_test_loss)
    print('Label classifier output layer Test loss:', label_clf_test_loss)
    print('Discriminator output layer Test accuracy:', discr_test_acc)
    print('Label classifier output layer Test accuracy:', label_clf_test_acc)


    if plot:
        plot_mist(x_train, y_train, 9, save_file_path='plots/test.png')


if __name__ == '__main__':
    main()