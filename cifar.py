from matplotlib import pyplot as plt
import numpy as np
import os
import pdb
from keras.datasets import cifar10


class CIFAR(object):
    """
    Prepare data batches for training and testing.

    Data files are from http://yann.lecun.com/exdb/mnist/

    The training set contains 60000 examples, and the test set 10000 examples.
    The first 5000 examples of the test set are taken from the original NIST training set.
    The last 5000 are taken from the original NIST test set.
    The first 5000 are cleaner and easier than the last 5000.
    each image is 28x28 pixels
    """
    def __init__(self):
        self.training_images = None
        self.training_labels = None

        self.testing_images = None
        self.testing_labels = None

    @staticmethod
    def to_catagorical(one_hot_labels):
        return np.nonzero(one_hot_labels)[1]


    def get_batch(self, batch_size, dataset='training', classes=[]):
        """
        get a batch of images and corresponding labels.

        returned images would have the shape of (batch_size, 28, 28);
        returned labels would have the shape of (batch_size, 10)

        :param batch_size:
        :param dataset: 'training' or 'testing'
        """
        if dataset == 'training':
            if self.training_images is None or self.training_labels is None:
                (self.training_images, self.training_labels), (_, _) = cifar10.load_data()
            images = self.training_images
            labels = self.training_labels
        elif dataset == 'testing':
            if self.testing_images is None or self.testing_labels is None:
                (_, _), (self.testing_images, self.testing_labels) = cifar10.load_data()
            images = self.testing_images
            labels = self.testing_labels
        else:
            return

        num_samples = labels.shape[0]
        if classes != []:
            catagorical_labels = self.to_catagorical(labels)
            good_classes = np.isin(catagorical_labels, classes)
            loc_of_good_classes = np.nonzero(good_classes)[0]
            idx = np.random.randint(loc_of_good_classes.shape[0], size=batch_size)
            idx = loc_of_good_classes[idx]
        else:
            idx = np.random.randint(num_samples, size=batch_size)


        return images[idx], labels[idx]


def main():
    cifar = CIFAR()
    images, labels = cifar.get_batch(10, 'training')

    for im, lb in zip(images, labels):
        plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
        plt.text(1, 1, lb, color='w')
        plt.show()


if __name__ == '__main__':
    main()
