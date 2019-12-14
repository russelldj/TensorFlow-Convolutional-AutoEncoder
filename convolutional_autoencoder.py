import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from models import *
from mnist import MNIST  # this is the MNIST data manager that provides training/testing batches
import pdb
import cv2

import lime
from lime import lime_image
import skimage.segmentation as seg
import sklearn.manifold

class ConvolutionalAutoencoder(object):
    """

    """
    def __init__(self):
        """
        build the graph
        """
        # place holder of input data
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])  # [#batch, img_height, img_width, #channels]

        # encode
        conv1 = Convolution2D([5, 5, 1, 32], activation=tf.nn.relu, scope='conv_1')(x)
        pool1 = MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool_1')(conv1)
        conv2 = Convolution2D([5, 5, 32, 32], activation=tf.nn.relu, scope='conv_2')(pool1)
        pool2 = MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool_2')(conv2)
        unfold = Unfold(scope='unfold')(pool2)
        encoded = FullyConnected(20, activation=tf.nn.relu, scope='encode')(unfold)
        # decode
        decoded = FullyConnected(7*7*32, activation=tf.nn.relu, scope='decode')(encoded)
        fold = Fold([-1, 7, 7, 32], scope='fold')(decoded)
        unpool1 = UnPooling((2, 2), output_shape=tf.shape(conv2), scope='unpool_1')(fold)
        deconv1 = DeConvolution2D([5, 5, 32, 32], output_shape=tf.shape(pool1), activation=tf.nn.relu, scope='deconv_1')(unpool1)
        unpool2 = UnPooling((2, 2), output_shape=tf.shape(conv1), scope='unpool_2')(deconv1)
        reconstruction = DeConvolution2D([5, 5, 1, 32], output_shape=tf.shape(x), activation=tf.nn.sigmoid, scope='deconv_2')(unpool2)

        # loss function
        loss = tf.nn.l2_loss(x - reconstruction)  # L2 loss

        # training
        training = tf.train.AdamOptimizer(1e-4).minimize(loss)

        #
        self.x = x
        self.encoded = encoded
        self.reconstruction = reconstruction
        self.loss = loss
        self.training = training

    def train(self, batch_size, passes, new_training=True, classes=list(range(10))):
        """

        :param batch_size:
        :param passes:
        :param new_training:
        :return:
        """
        mnist = MNIST()
        with tf.Session() as sess:
            # prepare session
            if new_training:
                saver, global_step = Model.start_new_session(sess)
            else:
                saver, global_step = Model.continue_previous_session(sess, ckpt_file='saver/checkpoint')

            # start training
            for step in range(1+global_step, 1+passes+global_step):
                x, y = mnist.get_batch(batch_size, classes=classes)
                self.training.run(feed_dict={self.x: x})

                if step % 10 == 0:
                    loss = self.loss.eval(feed_dict={self.x: x})
                    print("pass {}, training loss {}".format(step, loss))

                if step % 100 == 0:  # save weights
                    saver.save(sess, 'saver/cnn', global_step=step)
                    print('checkpoint saved')

    def weights_to_grid(self, weights, rows, cols):
        """convert the weights tensor into a grid for visualization"""
        height, width, in_channel, out_channel = weights.shape
        padded = np.pad(weights, [(1, 1), (1, 1), (0, 0), (0, rows * cols - out_channel)],
                        mode='constant', constant_values=0)
        transposed = padded.transpose((3, 1, 0, 2))
        reshaped = transposed.reshape((rows, -1))
        grid_rows = [row.reshape((-1, height + 2, in_channel)).transpose((1, 0, 2)) for row in reshaped]
        grid = np.concatenate(grid_rows, axis=0)

        return grid.squeeze()


    def reconstruct(self, vis_weights=True, classes="all"):
        """

        """
        def weights_to_grid(weights, rows, cols):
            """convert the weights tensor into a grid for visualization"""
            height, width, in_channel, out_channel = weights.shape
            padded = np.pad(weights, [(1, 1), (1, 1), (0, 0), (0, rows * cols - out_channel)],
                            mode='constant', constant_values=0)
            transposed = padded.transpose((3, 1, 0, 2))
            reshaped = transposed.reshape((rows, -1))
            grid_rows = [row.reshape((-1, height + 2, in_channel)).transpose((1, 0, 2)) for row in reshaped]
            grid = np.concatenate(grid_rows, axis=0)

            return grid.squeeze()

        mnist = MNIST()

        with tf.Session() as sess:
            saver, global_step = Model.continue_previous_session(sess, ckpt_file='saver/checkpoint')

            # visualize weights
            if vis_weights:
                first_layer_weights = tf.get_default_graph().get_tensor_by_name("conv_1/kernel:0").eval()
                grid_image = weights_to_grid(first_layer_weights, 4, 8)

                fig, ax0 = plt.subplots(ncols=1, figsize=(8, 4))
                ax0.imshow(grid_image, cmap=plt.cm.gray, interpolation='nearest')
                ax0.set_title('first conv layers weights')
                plt.show()

            # visualize results
            batch_size = 36
            x, y = mnist.get_batch(batch_size, dataset='testing')
            org, recon = sess.run((self.x, self.reconstruction), feed_dict={self.x: x})

            diff = (org - recon).squeeze()
            magnitude = np.linalg.norm(org.squeeze(), axis=(1, 2))
            error = np.linalg.norm(diff, axis=(1, 2))
            print(np.mean(error))
            error = error / magnitude

            input_images = weights_to_grid(org.transpose((1, 2, 3, 0)), 6, 6)
            recon_images = weights_to_grid(recon.transpose((1, 2, 3, 0)), 6, 6)
            errors = np.reshape(error, (6, 6))

            fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(10, 5))
            fig.suptitle("Trained with classes: {}".format(classes))
            ax0.imshow(input_images, cmap=plt.cm.gray, interpolation='nearest')
            ax0.set_title('input images')
            ax1.imshow(recon_images, cmap=plt.cm.gray, interpolation='nearest')
            ax1.set_title('reconstructed images')
            shown = ax2.imshow(errors, cmap=plt.cm.inferno, interpolation='nearest')
            plt.colorbar(shown, ax=ax2)
            ax2.set_title('Errors')
            plt.show()

    def compute_error(self, input_image, expand_dims=False):
        print(input_image.shape)
        input_image = input_image[...,0:1]
        if expand_dims:
            input_image = np.expand_dims(input_image, axis=0)
        with tf.Session() as sess:
            saver, global_step = Model.continue_previous_session(sess, ckpt_file='saver/checkpoint')
            org, recon = sess.run((self.x, self.reconstruction), feed_dict={self.x: input_image})
        error = np.linalg.norm(org - recon)
        #pdb.set_trace()
        class_probs =[self.return_probs(x, y) for x, y in zip(org, recon)]
        return class_probs

    def get_features(self, input_image, expand_dims=False):
        print(input_image.shape)
        input_image = input_image[...,0:1]
        if expand_dims:
            input_image = np.expand_dims(input_image, axis=0)
        with tf.Session() as sess:
            saver, global_step = Model.continue_previous_session(sess, ckpt_file='saver/checkpoint')
            feature = sess.run((self.encoded), feed_dict={self.x: input_image})

        return feature


    def test_features(self, classes=[0]):
        mnist = MNIST()
        images, labels = mnist.get_batch(3000, dataset='testing', classes=classes)
        labels = np.nonzero(labels)[1]
        features = self.get_features(images)
        tsne = sklearn.manifold.TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(features)
        for i in classes:
            matching = labels == i
            plt.scatter(tsne_results[matching, 0], tsne_results[matching, 1])
        plt.legend([str(i) for i in classes])
        plt.show()


    def return_probs(self, org, recon):
        error = np.linalg.norm(org - recon)
        scaled = min(error / 10.0, 1)
        return [1 - scaled, scaled]


    def test_compute_error(self, classes):

        mnist = MNIST()
        xin, _ = mnist.get_batch(30, dataset='testing', classes=classes)
        xout, _ = mnist.get_batch(6, dataset='testing', classes=[x for x in range(10) if x not in classes])

        x = np.concatenate((xin, xout))
        np.random.shuffle(x)
        errors = []
        for img in x:
            errors.append(self.compute_error(img, expand_dims=True)[0][1]) # prob of anomalous
        top_k_error = np.sort(errors)[-6]
        errors = np.reshape(errors, (6,6))


        input_images = self.weights_to_grid(x.transpose((1, 2, 3, 0)), 6, 6)

        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(10, 5))
        fig.suptitle("Trained with classes")
        ax0.imshow(input_images, cmap=plt.cm.gray, interpolation='nearest')
        ax0.set_title('input images')
        shown = ax1.imshow(errors, cmap=plt.cm.inferno, interpolation='nearest')
        plt.colorbar(shown, ax=ax1)
        ax2.imshow(errors >= top_k_error, plt.cm.inferno, interpolation="nearest")
        ax1.set_title('Errors')

        plt.show()



def lime(model):

    #process image
    mnist = MNIST()
    image, label = mnist.get_batch(1, dataset='testing')
    image = image[0]
    image = np.concatenate((image, image, image), axis=2)
    print(image.shape)
    print('Print image:')

    plt.imshow(np.squeeze(image) / 2 +0.5)
    plt.show()

    print(model.compute_error(image, expand_dims=True))

    #Explain
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image, model.compute_error, top_labels=2)
    print(explanation)

    #Show superpixels
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=True)
    #pdb.set_trace()
    print('Print superpixels:')
    marked = seg.mark_boundaries(temp/ 2 +0.5, mask).astype(np.uint8)
    plt.imshow(marked)
    plt.show()

def main():
    CLASSES = [0]
    EPOCS   = 1000
    TRAIN   = True
    conv_autoencoder = ConvolutionalAutoencoder()
    #lime(conv_autoencoder)
    if TRAIN:
        conv_autoencoder.train(batch_size=100, passes=EPOCS, new_training=True, classes=CLASSES)
    conv_autoencoder.test_features(CLASSES)
    pdb.set_trace()
    #conv_autoencoder.get_features()
    conv_autoencoder.test_compute_error(classes=CLASSES)
    error = conv_autoencoder.compute_error(np.zeros((28,28,1), dtype=np.uint8), expand_dims=True)
    print('error', error)
    #conv_autoencoder.reconstruct(False, classes=CLASSES)


if __name__ == '__main__':
    main()
