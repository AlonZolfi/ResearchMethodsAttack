from __future__ import absolute_import, division, print_function, unicode_literals

from keras.models import load_model

from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier

from train_model import load_data_set, evaluate_model


# Craft adversarial samples with FGSM
def fgsm_attack(classifier, x_test):
    print("Starting attack")
    epsilon = .1  # Maximum perturbation
    adv_crafter = FastGradientMethod(classifier, eps=epsilon)
    x_test_adv = adv_crafter.generate(x=x_test)
    print("Finished attack")
    return x_test_adv


def main():
    x_train, y_train, x_test, y_test, min, max = load_data_set()
    model = load_model('weights/weights.h5')
    classifier = KerasClassifier(model=model)
    evaluate_model(classifier, x_test, y_test)
    x_test_adv = fgsm_attack(classifier, x_test)
    evaluate_model(classifier, x_test_adv, y_test)


if __name__ == "__main__":
    main()
