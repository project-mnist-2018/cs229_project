""" Convolutional Neural Network MNIST classifier """
import argparse
from utils.misc import get_real_mnist, get_gan_mnist, plot_mist
from utils.preprocessing import preprocess_raw_mnist_data
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam
from vis.utils import utils
from keras import activations
from keras.initializers import glorot_uniform
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
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax, name='preds')
    ])

    #classifier.compile(optimizer=tf.train.AdamOptimizer(),
    #                   loss='sparse_categorical_crossentropy',
    #                   metrics=['accuracy'])
    classifier.compile(optimizer=keras.optimizers.Adam(),
                        loss=keras.losses.sparse_categorical_crossentropy,
                        metrics=['accuracy'])

    return classifier

#Can't acccess Tensorflow's optimizer attributes after instantiation
def cnn_keras_classifier():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=5, strides=1, activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax', name='preds'))
    return model

#Attention visualization
def attention_visualization(cnn_clf, x_test, y_test):
    #Visualizing saliency
    num_classes = 10
    y_test_cate = keras.utils.to_categorical(y_test, num_classes)
    class_idx = 0
    indices = np.where(y_test_cate[:, class_idx] == 1.)[0]

    # pick some random input from here.
    idx = indices[0]

    # Utility to search for layer index by name. 
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = utils.find_layer_idx(cnn_clf, 'preds')

    # Swap softmax with linear because 
    #It does not work! The reason is that maximizing an output node can be done by minimizing other outputs. 
    #Softmax is weird that way. It is the only activation that depends on other node output(s) in the layer
    cnn_clf.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(cnn_clf)

    #Show negation
    grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, seed_input=x_test[idx], 
                               backprop_modifier='guided', grad_modifier='negate')
    plt.imshow(grads, cmap='jet')
    plt.savefig("output/conv_4layer_negated.png")
    plt.show()

    #Attention saliency visualization
    for class_idx in np.arange(10):    
        indices = np.where(y_test_cate[:, class_idx] == 1.)[0]
        idx = indices[0]

        f, ax = plt.subplots(1, 4)
        ax[0].imshow(x_test[idx][..., 0])
        
        for i, modifier in enumerate([None, 'guided', 'relu']):
            grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, 
                                       seed_input=x_test[idx], backprop_modifier=modifier)
            if modifier is None:
                modifier = 'vanilla'
            ax[i+1].set_title(modifier)    
            ax[i+1].imshow(grads, cmap='jet')

        # TODO make sure that output folder structure is correct
        plt.savefig("output/attention_saliency/conv_4layer_multiple_modifiers_saliency_" + str(class_idx) + ".png")
        plt.show()

    #Attention CAM visualization
    # This corresponds to the Dense linear layer.
    for class_idx in np.arange(10):    
        indices = np.where(y_test_cate[:, class_idx] == 1.)[0]
        idx = indices[0]

        f, ax = plt.subplots(1, 4)
        ax[0].imshow(x_test[idx][..., 0])
        
        for i, modifier in enumerate([None, 'guided', 'relu']):
            grads = visualize_cam(model, layer_idx, filter_indices=class_idx, 
                                  seed_input=x_test[idx], backprop_modifier=modifier)        
            if modifier is None:
                modifier = 'vanilla'
            ax[i+1].set_title(modifier)
            ax[i+1].imshow(grads, cmap='jet')
        plt.savefig("output/attention_CAM/conv_4layer_multiple_modifiers_CAM_" + str(class_idx) + ".png")        
        plt.show()

    #model summary
    print(model.summary())


def main(plot=False, train=False, epochs=5, attention=False):
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

    if train:
        # Train classifier
        print('\ntrain the classifier')
        # cnn_clf.compile(optimizer=keras.optimizers.Adam(),
        #                 loss=keras.losses.sparse_categorical_crossentropy,
        #                 metrics=['accuracy'])

        #history = cnn_clf.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        history = cnn_clf.fit(x_train, y_train, epochs=epochs, validation_split=0.1)

        # Save weights
        cnn_clf.save_weights('weights/cnn_clf_4layer_%s.h5' % epochs)

        #Plots train and validation datasets
        #Get data from history
        print(history.history.keys())
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title("model accuracy")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("output/conv_4layer_model_accuracy.png")
        plt.show()
        #Save the plot

        #Plot the loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("output/conv_4layer_model_loss.png")
        plt.show()

    else:
        # Load the model weights
        import os
        weights_file_path = os.path.abspath(os.path.join(os.curdir, 'weights/cnn_clf_4layer_%s.h5' % epochs))
        if not print(os.path.exists(weights_file_path)):
            print("The weights file path specified does not exists: %s" % os.path.exists(weights_file_path))
        cnn_clf.load_weights(weights_file_path)

    print('\ntest the classifier')
    test_loss, test_acc = cnn_clf.evaluate(x_test[:1000], y_test[:1000])

    print('\n#######################################')
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    print('\ntest the classifier on gan mnist')
    test_loss, test_acc = cnn_clf.evaluate(x_gan_test[:1000], y_gan_test[:1000])

    print('\n#######################################')
    print('Test loss gan:', test_loss)
    print('Test accuracy gan:', test_acc)

    if attention:
        attention_visualization(cnn_clf, x_test, y_test)

    if plot:
        plot_mist(x_train, y_train, 9, save_file_path='plots/test.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", help="It specifies if you want to train the model first. (Default False)",
                        action="store_true",
                        dest='train')
    parser.add_argument("--epochs",
                        "-e",
                        help=(
                            "It specifies how many epochs to use for training the "
                            "model or which corresponding save weights to use. (Default 5)"
                        ),
                        default='5',
                        type=int,
                        dest='epochs'
                        )
    parser.add_argument("--attention",
                        "-a",
                        help=(
                            "It specifies if you want to plot the attention visualization at the end (Default False)"
                        ),
                        action="store_true",
                        dest='attention'
                        )
    args = parser.parse_args()
    main(train=args.train, epochs=args.epochs, attention=args.attention)
