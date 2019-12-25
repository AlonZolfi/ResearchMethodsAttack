from __future__ import absolute_import, division, print_function, unicode_literals

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

import numpy as np

from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset

from tensorflow.python.client import device_lib
if str(device_lib.list_local_devices()).__contains__('GPU'):
    import keras
    import tensorflow as tf
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    cuda_cores = 768  # GTX 1050 Ti
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': cuda_cores})
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)


# Read MNIST
def load_data_set():
    (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('mnist'))
    return x_train, y_train, x_test, y_test, min_, max_


# Create Keras convolutional neural network - basic architecture from Keras examples
# Source here: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
def train_model(x_train, y_train, min_, max_):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1:][0], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    check_point = ModelCheckpoint("weights/weights.h5", monitor='val_loss', save_best_only=False)
    callback = [check_point]
    classifier = KerasClassifier(model=model, clip_values=(min_, max_))
    classifier.fit(x_train, y_train, nb_epochs=20, batch_size=128, callbacks=callback)
    return classifier


# Evaluate the classifier on the test set
def evaluate_model(classifier, x_test, y_test):
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy: %.2f%%" % (acc * 100))


# Craft adversarial samples with FGSM
def fgsm_attack(classifier, x_test):
    epsilon = .1  # Maximum perturbation
    adv_crafter = FastGradientMethod(classifier, eps=epsilon)
    x_test_adv = adv_crafter.generate(x=x_test)
    return x_test_adv


# Evaluate the classifier on the adversarial examples
def evaluate_attack(classifier, x_test_adv, y_test):
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy on adversarial sample: %.2f%%" % (acc * 100))


def main():
    model_type = "train"  # train or load
    x_train, y_train, x_test, y_test, min, max = load_data_set()
    if model_type == "train":
        classifier = train_model(x_train, y_train, min, max)
    elif model_type == "load":
        model = load_model('weights/weights.h5')
        classifier = KerasClassifier(model=model)
    evaluate_model(classifier, x_test, y_test)
    x_test_adv = fgsm_attack(classifier, x_test)
    evaluate_attack(classifier, x_test_adv, y_test)


if __name__ == "__main__":
    main()
