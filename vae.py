import tensorflow as tf

from layers import convLayer, tConvLayer, fcLayer

class VAE(object):

	def __init__(self, x, n_z):
		
		self.n_z = n_z
		self.X = x
		X_reshaped = tf.reshape(self.X, [-1, 28, 28, 1])

		z_mean, z_stddev = self.recognition(X_reshaped)
		epsilon = tf.random_normal([self.X.get_shape()[0], self.n_z], 0, 1, dtype=tf.float32)
		z_generated = z_mean + (tf.exp(z_stddev) * epsilon)

		self.generated_images = self.generation(z_generated)

		generated_flatten = tf.contrib.layers.flatten(self.generated_images)

		self.reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = generated_flatten, labels = self.X), 1)
		self.latent_loss = 0.5 * tf.reduce_sum(1 + 2.0*z_stddev - tf.square(z_mean) - tf.exp(2.0*z_stddev) ,1)
		self.cost = tf.reduce_mean(self.reconstruction_loss + self.latent_loss)

	def recognition(self, input_image):

		with tf.variable_scope("recognition"):
			conv1 = convLayer(input_image, 5, 5, 64, "conv1")
			conv2 = convLayer(conv1, 5, 5, 128, "conv2")
			conv2_flat = tf.contrib.layers.flatten(conv2)

			mean = fcLayer(conv2_flat, 20, "mean")
			stddev = fcLayer(conv2_flat, 20, "stddev")

		return mean, stddev

	def generation(self, z):

		with tf.variable_scope("generation"):
			z_expand = tf.reshape(fcLayer(z, 7*7*128, "z_expand"), [-1, 7, 7, 128])

			tconv1 = tConvLayer(z_expand, 5, 5, [int(z_expand.get_shape()[0]), 14, 14, 64], "tconv1")
			tconv2 = tConvLayer(tconv1, 5, 5, [int(tconv1.get_shape()[0]), 28, 28, 1], "tconv2")

		return tconv2