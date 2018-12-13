""" Convolutional Neural Network MNIST classifier """
import itertools
from utils.misc import get_real_mnist, get_gan_mnist, plot_mist
from utils.preprocessing import preprocess_raw_mnist_data
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam
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
        keras.layers.Dense(10, activation=tf.nn.softmax, name='preds')
    ])

    #classifier.compile(optimizer=tf.train.AdamOptimizer(),
    #                   loss='sparse_categorical_crossentropy',
    #                   metrics=['accuracy'])
    classifier.compile(optimizer=keras.optimizers.Adam(),
                        loss=keras.losses.sparse_categorical_crossentropy,
                        metrics=['accuracy'])

    return classifier

#Attention visualization
def attention_visualization(cnn_clf, x_test, y_test, epoch, img):
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
    plt.savefig("output/conv_5layer_negated.png")
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
    
        plt.savefig("output/attention_saliency/conv_5layer_" + img +"_epoch_" + str(epoch) + "_multiple_modifiers_saliency_" + str(class_idx) + ".png")        
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
        plt.savefig("output/attention_CAM/conv_5layer_" + img + "_epoch_" + str(epoch) + "_multiple_modifiers_CAM_" + str(class_idx) + ".png")        
        plt.show()

    #model summary
    print(model.summary())

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

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

    epochs = 100

    if train:
        # Train classifier
        print('\ntrain the classifier')

        #history = cnn_clf.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        history = cnn_clf.fit(x_train, y_train, epochs=epochs, validation_split=0.1)

        # Save weights
        cnn_clf.save_weights('weights/cnn_clf_5layer_%s.h5' % epochs)

        #Plots train and validation datasets
        #Get data from history
        print(history.history.keys())
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title("model accuracy")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("output/conv_5layer_epoch_" + str(epochs) + "_model_accuracy.png")
        plt.show()
        #Save the plot

        #Plot the loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("output/conv_5layer_epoch_" + str(epochs) + "model_loss.png")
        plt.show()

    else:
        # Load the model weights
        import os
        weights_file_path = os.path.abspath(os.path.join(os.curdir, 'weights/cnn_clf_5layer_%s.h5' % epochs))
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

    attention_visualization(cnn_clf, x_test[:1000], y_test[:1000], epochs, "real")
    attention_visualization(cnn_clf, x_gan_test[:1000], y_gan_test[:1000], epochs, "synthetic")

    from sklearn.metrics import confusion_matrix

    #Original predict returns 1 hot encoding, so use argmax instead
    y_pred = np.argmax(cnn_clf.predict(x_test[:1000]), axis=1)
    #y_pred = tf.argmax(cnn_clf.predict(x_test), axis=1)
    #cm = tf.confusion_matrix(y_test, y_pred, num_classes=10)
    
    #sess = tf.Session()
    #with sess.as_default():
    #    print(sess.run(cm))
    cm = confusion_matrix(y_test[:1000], y_pred)
    class_names= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    y_gan_pred = np.argmax(cnn_clf.predict(x_gan_test[:1000]), axis=1)
    gan_cm = confusion_matrix(y_gan_test[:1000], y_gan_pred)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=class_names,
                          title='Confusion matrix real images, without normalization')
    plt.savefig("output/confusion_matrix_real_cnn_5layer_epoch_" + str(epochs) + ".png")

    plt.figure()
    plot_confusion_matrix(gan_cm, classes=class_names,
                          title='Confusion matrix synthetic images, without normalization')
    plt.savefig("output/confusion_matrix_synthetic_cnn_5layer_epoch_" + str(epochs) + ".png")

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=class_names, normalize=True,
                          title='Normalized confusion matrix real images')
    plt.savefig("output/confusion_matrix_real_normalized_cnn_5layer_epoch_" + str(epochs) + ".png")

    plt.figure()
    plot_confusion_matrix(gan_cm, classes=class_names, normalize=True,
                          title='Normalized confusion matrix synthetic images')
    plt.savefig("output/confusion_matrix_synthetic_normalized_cnn_5layer_epoch_" + str(epochs) + ".png")

    plt.show()

    if plot:
        plot_mist(x_train, y_train, 9, save_file_path='plots/test.png')


if __name__ == '__main__':
    main(train=False)
