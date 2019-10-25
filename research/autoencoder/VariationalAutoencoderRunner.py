from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from autoencoder_models.VariationalAutoencoder import VariationalAutoencoder
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def min_max_scale(X_train, X_test):
    preprocessor = prep.MinMaxScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


X_train, X_test = min_max_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 8
batch_size = 128
display_step = 1

autoencoder = VariationalAutoencoder(
    n_input=784,
    n_hidden=200,
    optimizer=tf.train.AdamOptimizer(learning_rate = 0.001))

# Instantiate a SummaryWriter to output summaries and the Graph.
log_dir = os.path.join('.', 'logs')
autoencoder.writeSummary(log_dir)

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data
        cost = autoencoder.partial_fit(batch_xs)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

        if i % 10 == 0:
            n = 30
            plt.figure(figsize=(20, 4))
            X_test2 = autoencoder.reconstruct(X_test)
            for j in range(n):
                ax = plt.subplot(2, n, j + 1)
                plt.imshow(X_test2[j].reshape(28, 28))
                # plt.imshow(X_test[i].reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.savefig('test_' + str(epoch + 1).zfill(3) + '_' + str(i).zfill(5) + '.png')

    # Display logs per epoch step
    if epoch % display_step == 0:
        # Update the events file.
        # summary_str = sess.run(summary)
        # summary_writer.add_summary(summary_str, epoch)
        # summary_writer.flush()

        print("Epoch:", '%d,' % (epoch + 1),
              "Cost:", "{:.9f}".format(avg_cost))


        # n = 10
        # plt.figure(figsize=(20, 4))
        # X_test2 = autoencoder.reconstruct(X_test)
        # for i in range(n):
        #     ax = plt.subplot(2, n, i + 1)
        #     plt.imshow(X_test2[i].reshape(28, 28))
        #     plt.gray()
        #     ax.get_xaxis().set_visible(False)
        #     ax.get_yaxis().set_visible(False)
        # plt.savefig('result_' + str(epoch + 1).zfill(3) + '.png')
print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
