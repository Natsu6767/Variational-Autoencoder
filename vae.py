import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from layers import convLayer, tConvLayer, fcLayer

class VAE(object):
    """Creating the VAE model"""

    def __init__(self, x, n_z, batch_size):
        
        self.n_z = n_z
        self.X = x
        self.batch_size = batch_size
        X_reshaped = tf.reshape(self.X, [-1, 28, 28, 1])

        self.z_mean, z_stddev = self.recognition(X_reshaped)
        epsilon = tf.random_normal(tf.stack([self.batch_size, self.n_z]))
        self.z_generated = tf.add(self.z_mean, tf.multiply(tf.exp(z_stddev), epsilon))

        self.generated_images = self.generation(self.z_generated)
        generated_flatten = tf.reshape(self.generated_images, [self.batch_size, 28*28])
        
        #Mean Squared Error
        self.reconstruction_loss = tf.reduce_sum(tf.squared_difference(generated_flatten, self.X), 1)
        
        #Binary cross-entropy
        #self.reconstruction_loss = -tf.reduce_sum(self.X * tf.log(1e-8 + generated_flatten) + (1-self.X) * tf.log(1e-8 + 1 - generated_flatten),1)
        
        #KL Divergence loss term
        self.latent_loss = -0.5 * tf.reduce_sum(1 + 2.0*z_stddev - tf.square(self.z_mean) - tf.exp(2.0*z_stddev),1)
        self.cost = tf.reduce_mean(self.reconstruction_loss + self.latent_loss)
    
    #Encoder
    def recognition(self, input_image):

        with tf.variable_scope("recognition"):
            conv1 = convLayer(input_image, 5, 5, 32, "conv1")
            conv2 = convLayer(conv1, 5, 5, 64, "conv2")
            conv3 = convLayer(conv2, 3, 3, 64, "conv3", stride=1)
            conv3_flat = tf.contrib.layers.flatten(conv3)

            mean = fcLayer(conv3_flat, self.n_z, relu=False, name="mean")
            stddev = fcLayer(conv3_flat, self.n_z, relu=False, name="stddev")

        return mean, stddev
    
    #Decoder
    def generation(self, z):

        with tf.variable_scope("generation"):
            z_expand = tf.reshape(fcLayer(z, 4*4*64, "z_expand"), [-1, 4, 4, 64])
            
            tconv1 = tConvLayer(z_expand, 3, 3, [self.batch_size, 7, 7, 64], "tconv1")
            tconv2 = tConvLayer(tconv1, 5, 5, [self.batch_size, 14, 14, 32], "tconv2")
            tconv3 = tConvLayer(tconv2, 5, 5, [self.batch_size, 28, 28, 1], "tconv3", activation=False)

            out = tf.nn.sigmoid(tconv3)

        return out